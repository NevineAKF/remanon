"""
Band B contract tests — real core objects, no mocks of the core itself.

Contract B is served by the in-process mock engine through an httpx
ASGITransport: zero network, zero GPU. HTTP calls to /v1/chat/completions
are counted at the transport layer, so prefill counts are observed facts,
not internal assertions alone.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.budgeter import BudgetExceeded, MemoryBudgeter
from core.generator import CoreGenerator, NotMaterializedError
from core.materializer import LazyMaterializer, master_system_content
from core.memory_model import GIB, MemoryModel, ModelSpec
from core.metrics import CoreMetrics
from core.registry import Engine, EngineRegistry
from core.residency import PinnedBlockError, ResidencyManager
from deploy.mock_engine.main import app as mock_app

MODEL = "remanon-correlator-13b"


class CountingTransport(httpx.AsyncBaseTransport):
    """Wraps ASGITransport and counts real Contract B prefill calls."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner
        self.chat_calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            self.chat_calls += 1
        return await self._inner.handle_async_request(request)


@pytest.fixture()
def transport() -> CountingTransport:
    return CountingTransport(httpx.ASGITransport(app=mock_app))


@pytest.fixture()
def registry(transport: CountingTransport) -> EngineRegistry:
    reg = EngineRegistry(transport=transport)
    reg.register(Engine(model=MODEL, base_url="http://mock-engine", port=8000))
    return reg


def _pressure_model() -> MemoryModel:
    """usable = 64 − 4 = 60 GB; pinned model = 26 + 10 = 36 GB; delta room = 24 GB."""
    return MemoryModel(
        total_capacity_gb=64.0,
        headroom_gb=4.0,
        models={MODEL: ModelSpec(MODEL, weights_gb=26.0, master_gb=10.0)},
        agent_delta_budget_gb={"hunter": 100.0, "flood": 100.0, "b": 100.0},
    )


# ---------------------------------------------------------------------------
# (a) Pinned blocks survive memory pressure
# ---------------------------------------------------------------------------


class TestPinnedNeverEvicted:
    async def test_pinned_block_survives_full_budget_and_eviction_attempts(self) -> None:
        mm = _pressure_model()
        budgeter = MemoryBudgeter(mm)
        residency = ResidencyManager(
            on_release=lambda lease: budgeter.release_delta(lease.lease_id)
        )

        await budgeter.pin_model(MODEL)
        await residency.lease("ctx-master", MODEL, "hunter")  # block now pinned

        # Synthetic pressure: fill every remaining byte with deltas.
        await budgeter.admit_delta("flood-1", "flood", 24.0)
        assert budgeter.ledger.available_bytes == 0

        # New admissions fail loudly...
        with pytest.raises(BudgetExceeded):
            await budgeter.admit_delta("flood-2", "flood", 1.0)

        # ...and repeated eviction attempts on the pinned block always raise.
        for _ in range(3):
            with pytest.raises(PinnedBlockError):
                await residency.evict(MODEL)

        # Pinned partitions are untouched by the whole ordeal.
        snap = budgeter.ledger.snapshot()
        assert snap["weights_gb"] == 26.0
        assert snap["masters_gb"] == 10.0
        assert residency.has_block(MODEL)
        assert residency.is_pinned(MODEL)

    async def test_evict_unknown_block_raises_key_error(self) -> None:
        residency = ResidencyManager()
        with pytest.raises(KeyError):
            await residency.evict("no-such-model")


# ---------------------------------------------------------------------------
# (b) N concurrent lease+materialize → exactly one prefill
# ---------------------------------------------------------------------------


class TestSinglePrefill:
    async def test_ten_concurrent_calls_one_prefill(
        self, registry: EngineRegistry, transport: CountingTransport
    ) -> None:
        residency = ResidencyManager()
        materializer = LazyMaterializer(registry, transport=transport)

        async def worker(i: int):
            lease = await residency.lease("ctx-master", MODEL, agent=f"agent-{i}")
            handle = await materializer.materialize("ctx-master", MODEL)
            return lease, handle

        results = await asyncio.gather(*(worker(i) for i in range(10)))

        # Observed at the HTTP layer: exactly one prefill hit Contract B.
        assert transport.chat_calls == 1
        assert materializer.prefills_performed == 1
        assert materializer.prefills_avoided == 9

        # All ten callers share the same handle; all ten leases are active.
        handle_ids = {handle.handle_id for _, handle in results}
        assert len(handle_ids) == 1
        assert residency.active_leases == 10

    async def test_sequential_materialize_is_idempotent(
        self, registry: EngineRegistry, transport: CountingTransport
    ) -> None:
        materializer = LazyMaterializer(registry, transport=transport)
        h1 = await materializer.materialize("ctx-master", MODEL)
        h2 = await materializer.materialize("ctx-master", MODEL)
        assert h1 is h2
        assert transport.chat_calls == 1
        assert materializer.prefills_avoided == 1


# ---------------------------------------------------------------------------
# (c) Admission at the exact boundary
# ---------------------------------------------------------------------------


class TestAdmissionBoundary:
    @staticmethod
    def _boundary_model() -> MemoryModel:
        """usable = 192 − 12 = 180 GB; pinned = 100 + 40 = 140 GB; exact room = 40 GB."""
        return MemoryModel(
            total_capacity_gb=192.0,
            headroom_gb=12.0,
            models={MODEL: ModelSpec(MODEL, weights_gb=100.0, master_gb=40.0)},
            agent_delta_budget_gb={"a": 1000.0},
        )

    async def test_request_landing_exactly_at_boundary_is_admitted(self) -> None:
        budgeter = MemoryBudgeter(self._boundary_model())
        await budgeter.pin_model(MODEL)

        grant = await budgeter.admit_delta("k", "a", 40.0)  # lands exactly at capacity−headroom
        assert grant.granted_bytes == 40 * GIB
        assert not grant.shrunk
        assert budgeter.ledger.available_bytes == 0

    async def test_one_byte_over_boundary_is_rejected(self) -> None:
        budgeter = MemoryBudgeter(self._boundary_model())
        await budgeter.pin_model(MODEL)

        one_byte_gb = 1 / GIB
        with pytest.raises(BudgetExceeded):
            await budgeter.admit_delta("k", "a", 40.0 + one_byte_gb)
        # The failed admission must not leak partial accounting.
        assert budgeter.ledger.deltas_bytes == 0


# ---------------------------------------------------------------------------
# (d) Pressure resolution order: shrink → reject → masters never touched
# ---------------------------------------------------------------------------


class TestPressureOrder:
    async def test_shrink_then_reject_masters_intact(self) -> None:
        mm = _pressure_model()  # delta room = 24 GB
        budgeter = MemoryBudgeter(mm)
        await budgeter.pin_model(MODEL)

        await budgeter.admit_delta("d1", "flood", 20.0)  # 4 GB left

        # (a) shrink: full request 8 GB does not fit, min 2 GB does → granted 4 GB.
        grant = await budgeter.admit_delta("d2", "b", 8.0, min_gb=2.0)
        assert grant.shrunk
        assert grant.granted_bytes == 4 * GIB
        assert budgeter.ledger.available_bytes == 0

        # (b) reject: even the minimum no longer fits.
        with pytest.raises(BudgetExceeded):
            await budgeter.admit_delta("d3", "b", 1.0, min_gb=0.5)

        # (c) pinned partitions never touched at any point under pressure.
        snap = budgeter.ledger.snapshot()
        assert snap["weights_gb"] == 26.0
        assert snap["masters_gb"] == 10.0

    async def test_per_agent_delta_cap_shrinks_before_global_pressure(self) -> None:
        mm = MemoryModel(
            total_capacity_gb=64.0,
            headroom_gb=4.0,
            models={MODEL: ModelSpec(MODEL, weights_gb=26.0, master_gb=10.0)},
            agent_delta_budget_gb={"capped": 6.0},
        )
        budgeter = MemoryBudgeter(mm)
        await budgeter.pin_model(MODEL)

        # Global room is 24 GB but the agent cap is 6 GB → shrunk to 6.
        grant = await budgeter.admit_delta("d1", "capped", 10.0, min_gb=1.0)
        assert grant.granted_bytes == 6 * GIB


# ---------------------------------------------------------------------------
# (e) Lease release frees the delta budget
# ---------------------------------------------------------------------------


class TestLeaseReleaseFreesBudget:
    async def test_release_returns_delta_bytes_and_unpins(self) -> None:
        mm = _pressure_model()
        budgeter = MemoryBudgeter(mm)
        residency = ResidencyManager(
            on_release=lambda lease: budgeter.release_delta(lease.lease_id)
        )
        await budgeter.pin_model(MODEL)

        lease = await residency.lease("ctx-master", MODEL, "hunter")
        await budgeter.admit_delta(lease.lease_id, "hunter", 6.0)
        available_before = budgeter.ledger.available_bytes

        await residency.release(lease.lease_id)

        assert budgeter.ledger.available_bytes == available_before + 6 * GIB
        assert residency.active_leases == 0
        assert not residency.is_pinned(MODEL)

        # Unpinned now → eviction succeeds through the one legal code path.
        await residency.evict(MODEL)
        assert not residency.has_block(MODEL)

    async def test_release_unknown_lease_raises(self) -> None:
        residency = ResidencyManager()
        with pytest.raises(KeyError):
            await residency.release("no-such-lease")


# ---------------------------------------------------------------------------
# Engine registry health via Contract B (mock engine, in-process)
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    async def test_health_check_against_mock_engine(self, transport: CountingTransport) -> None:
        reg = EngineRegistry(transport=transport)
        reg.register(Engine(model=MODEL, base_url="http://mock-engine", port=8000))
        reg.register(Engine(model="ghost-model", base_url="http://mock-engine", port=8000))

        health = await reg.health_check()

        assert health[MODEL] is True
        assert health["ghost-model"] is False
        assert reg.resolve(MODEL).healthy is True
        assert reg.resolve(MODEL).last_checked is not None

    def test_resolve_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            EngineRegistry().resolve("nope")


# ---------------------------------------------------------------------------
# (f) Agents read through the master: generation carries the pinned prefix
# ---------------------------------------------------------------------------


class RecordingTransport(httpx.AsyncBaseTransport):
    """Wraps ASGITransport and captures every Contract B chat request body."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner
        self.chat_bodies: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            self.chat_bodies.append(json.loads(request.content))
        return await self._inner.handle_async_request(request)


def _read_through_stack() -> tuple[RecordingTransport, LazyMaterializer, CoreGenerator]:
    transport = RecordingTransport(httpx.ASGITransport(app=mock_app))
    registry = EngineRegistry(transport=transport)
    registry.register(Engine(model=MODEL, base_url="http://mock-engine", port=8000))
    materializer = LazyMaterializer(registry, transport=transport)
    generator = CoreGenerator(registry, materializer, transport=transport)
    return transport, materializer, generator


class TestAgentsReadThroughMaster:
    async def test_generate_carries_exact_master_prefix(self) -> None:
        transport, materializer, generator = _read_through_stack()
        await materializer.materialize("ctx-master", MODEL, context_text="X")

        await generator.generate("correlator", MODEL, "role: correlator", "case body")

        # Observed on the wire: chat_bodies[0] is the prefill, [1] the generate.
        assert len(transport.chat_bodies) == 2
        system = transport.chat_bodies[1]["messages"][0]
        assert system["role"] == "system"
        assert system["content"].startswith(master_system_content("ctx-master", "X"))

    async def test_two_agents_same_model_share_byte_identical_prefix(self) -> None:
        transport, materializer, generator = _read_through_stack()
        await materializer.materialize("ctx-master", MODEL, context_text="X")

        await generator.generate("correlator", MODEL, "role: correlator", "case body")
        await generator.generate("reporter", MODEL, "role: reporter", "case body")

        prefix = master_system_content("ctx-master", "X")
        first = transport.chat_bodies[1]["messages"][0]["content"]
        second = transport.chat_bodies[2]["messages"][0]["content"]
        # Byte-identical through the master prefix...
        assert first[: len(prefix)] == prefix
        assert second[: len(prefix)] == prefix
        # ...and diverging only after it (the role prompts differ).
        assert first[len(prefix) :] != second[len(prefix) :]

    async def test_generate_before_materialize_still_raises(self) -> None:
        _, _, generator = _read_through_stack()
        with pytest.raises(NotMaterializedError):
            await generator.generate("correlator", MODEL, "role: correlator", "case body")


# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------


class TestMetrics:
    async def test_export_reflects_sharing_and_ledger(
        self, registry: EngineRegistry, transport: CountingTransport
    ) -> None:
        mm = MemoryModel()  # default model set: correlator-13b master = 10 GB
        budgeter = MemoryBudgeter(mm)
        residency = ResidencyManager()
        materializer = LazyMaterializer(registry, transport=transport)
        metrics = CoreMetrics(
            memory_model=mm,
            materializer=materializer,
            residency=residency,
            budgeter=budgeter,
        )

        await budgeter.pin_model(MODEL)
        # correlator and reporter share the correlator-13b master.
        await residency.lease("ctx-master", MODEL, "correlator")
        await residency.lease("ctx-master", MODEL, "reporter")
        await materializer.materialize("ctx-master", MODEL)
        await materializer.materialize("ctx-master", MODEL)  # avoided

        exported = metrics.export()

        assert exported["prefills_performed"] == 1
        assert exported["prefills_avoided"] == 1
        assert exported["active_leases"] == 2
        # (2 agents sharing − 1) × 10 GB master
        assert exported["gb_saved_vs_per_agent"] == 10.0
        assert exported["ledger"]["masters_gb"] == 10.0
        assert exported["ledger"]["weights_gb"] == 26.0
        assert isinstance(exported["ledger"]["available_bytes"], int)
