"""
Layer L8 orchestrator tests — no network.

Contract B is the in-process mock engine via httpx ASGITransport. The five
agents are real; only failure-injection tests substitute a subclass.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx
import jsonschema
import pytest

from app.adapter.digest import DigestBuilder
from app.agents.base import BaseAgent
from app.agents.correlator_agent import CorrelatorAgent
from app.agents.hunter_agent import HunterAgent
from app.agents.reporter_agent import ReporterAgent
from app.agents.topology_agent import TopologyAgent
from app.agents.triage_agent import TriageAgent
from app.dataplane.normalizer import HDFSNormalizer, TelemetryRecord
from app.dataplane.parser import parse_file
from app.dataplane.store import TelemetryStore
from app.orchestrator.orchestrator import BurstDetector, EventLog, Orchestrator
from contracts.contract_a import Artifact, _load_schema
from core.generator import CoreGenerator
from core.materializer import LazyMaterializer
from core.registry import EngineRegistry, default_engines
from core.residency import ResidencyManager
from deploy.mock_engine.main import app as mock_app

_FIXTURE = Path("tests/fixtures/hdfs_sample.log")


class DelayTransport(httpx.AsyncBaseTransport):
    """Adds a fixed delay before each request so concurrency is observable."""

    def __init__(self, inner: httpx.AsyncBaseTransport, delay_s: float) -> None:
        self._inner = inner
        self._delay_s = delay_s

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self._delay_s)
        return await self._inner.handle_async_request(request)


@dataclass
class Runtime:
    residency: ResidencyManager
    materializer: LazyMaterializer
    generator: CoreGenerator
    agents: dict[str, BaseAgent]
    orchestrator: Orchestrator


def build_runtime(
    transport: httpx.AsyncBaseTransport,
    store: TelemetryStore | None = None,
    *,
    worker_timeout_s: float = 10.0,
    agent_overrides: dict[str, BaseAgent] | None = None,
) -> Runtime:
    registry = EngineRegistry(transport=transport)
    for engine in default_engines("http://mock-engine"):
        registry.register(engine)
    residency = ResidencyManager()
    materializer = LazyMaterializer(registry, transport=transport)
    generator = CoreGenerator(registry, materializer, transport=transport)
    deps: dict = {"residency": residency, "materializer": materializer, "generator": generator}
    agents: dict[str, BaseAgent] = {
        "triage": TriageAgent(**deps),
        "correlator": CorrelatorAgent(**deps),
        "hunter": HunterAgent(store=store, **deps),
        "topology": TopologyAgent(**deps),
        "reporter": ReporterAgent(**deps),
    }
    if agent_overrides:
        agents.update(agent_overrides)
    orchestrator = Orchestrator(agents, EventLog(), worker_timeout_s=worker_timeout_s)
    return Runtime(residency, materializer, generator, agents, orchestrator)


def make_case(case_id: str = "case-test-1") -> dict:
    return {
        "case_id": case_id,
        "opened_at": "2008-11-09T20:35:23+00:00",
        "record_count": 3,
        "trigger_records": [
            {
                "ts": "2008-11-09T20:35:23+00:00",
                "node": "10.250.19.102",
                "level": "ERROR",
                "component": "dfs.DataNode",
                "message": "Failed to transfer blk_-1608999687919862906",
            }
        ],
    }


@pytest.fixture()
def transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=mock_app)


# ---------------------------------------------------------------------------
# (a) Deterministic state order
# ---------------------------------------------------------------------------


class TestStateOrder:
    async def test_states_in_canonical_order(self, transport) -> None:
        rt = build_runtime(transport)
        case = make_case()
        result = await rt.orchestrator.run_case(case)

        states = [
            e.data["state"]
            for e in rt.orchestrator.event_log.events(case_id=case["case_id"], kind="state")
        ]
        assert states == ["INTAKE", "TRIAGE", "FAN_OUT", "JOIN", "REPORTER", "EMIT"]
        assert isinstance(result, Artifact)
        assert result.agent == "reporter"


# ---------------------------------------------------------------------------
# (b) The three workers overlap in time
# ---------------------------------------------------------------------------


class TestWorkerConcurrency:
    async def test_workers_overlap(self) -> None:
        slow = DelayTransport(httpx.ASGITransport(app=mock_app), delay_s=0.05)
        rt = build_runtime(slow)
        case = make_case("case-overlap")
        await rt.orchestrator.run_case(case)

        log = rt.orchestrator.event_log
        starts = {e.data["agent"]: e.ts for e in log.events(case["case_id"], "worker_start")}
        ends = {e.data["agent"]: e.ts for e in log.events(case["case_id"], "worker_end")}
        assert set(starts) == {"correlator", "hunter", "topology"}
        assert set(ends) == {"correlator", "hunter", "topology"}

        # Every worker started before any worker finished → all three were
        # simultaneously in flight.
        assert max(starts.values()) < min(ends.values())


# ---------------------------------------------------------------------------
# (c) Worker timeout → degraded artifact, pipeline completes
# ---------------------------------------------------------------------------


class TestWorkerTimeout:
    async def test_hanging_worker_degrades_and_pipeline_completes(self, transport) -> None:
        class HangingHunter(HunterAgent):
            async def run(self, case):  # noqa: ARG002
                await asyncio.sleep(30)
                raise AssertionError("unreachable")

        registry = EngineRegistry(transport=transport)
        for engine in default_engines("http://mock-engine"):
            registry.register(engine)
        residency = ResidencyManager()
        materializer = LazyMaterializer(registry, transport=transport)
        generator = CoreGenerator(registry, materializer, transport=transport)
        deps: dict = {
            "residency": residency,
            "materializer": materializer,
            "generator": generator,
        }
        rt = build_runtime(
            transport,
            worker_timeout_s=0.3,
            agent_overrides={"hunter": HangingHunter(**deps)},
        )
        case = make_case("case-timeout")
        result = await rt.orchestrator.run_case(case)

        assert isinstance(result, Artifact)
        assert result.agent == "reporter"
        degraded = rt.orchestrator.event_log.events(case["case_id"], "degraded")
        assert len(degraded) == 1
        assert degraded[0].data == {"agent": "hunter", "reason": "timeout"}


# ---------------------------------------------------------------------------
# (d) Invalid output → one retry, then inconclusive
# ---------------------------------------------------------------------------


class TestInvalidOutputRetry:
    async def test_two_failures_then_degraded(self, transport) -> None:
        calls = {"n": 0}

        class BrokenCorrelator(CorrelatorAgent):
            async def run(self, case):  # noqa: ARG002
                calls["n"] += 1
                raise ValueError("unparseable model output")

        rt_base = build_runtime(transport)
        broken = BrokenCorrelator(
            residency=rt_base.residency,
            materializer=rt_base.materializer,
            generator=rt_base.generator,
        )
        rt = build_runtime(transport, agent_overrides={"correlator": broken})
        case = make_case("case-invalid")
        result = await rt.orchestrator.run_case(case)

        assert calls["n"] == 2, "expected exactly one retry (two attempts total)"
        assert isinstance(result, Artifact)  # pipeline still completed

        log = rt.orchestrator.event_log
        attempts = [
            e
            for e in log.events(case["case_id"], "attempt_failed")
            if e.data["agent"] == "correlator"
        ]
        assert [a.data["attempt"] for a in attempts] == [1, 2]
        degraded = [
            e for e in log.events(case["case_id"], "degraded") if e.data["agent"] == "correlator"
        ]
        assert len(degraded) == 1
        assert degraded[0].data["reason"] == "invalid_output"


# ---------------------------------------------------------------------------
# (e) End-to-end on the fixture file
# ---------------------------------------------------------------------------


def _fixture_records() -> list[TelemetryRecord]:
    parsed = parse_file(_FIXTURE)
    normalizer = HDFSNormalizer()
    return [normalizer.normalize(p) for p in parsed.records]


class TestEndToEnd:
    async def test_burst_in_reporter_artifact_out(self, transport, tmp_path: Path) -> None:
        records = _fixture_records()
        store = TelemetryStore(tmp_path / "e2e.duckdb")
        store.write_records(records)

        detector = BurstDetector(threshold=3, window_s=60.0)
        cases = [case for r in sorted(records, key=lambda r: r.ts) if (case := detector.observe(r))]
        assert cases, "fixture burst should open at least one case"

        rt = build_runtime(transport, store=store)
        final = await rt.orchestrator.run_case(cases[0])

        assert isinstance(final, Artifact)
        assert final.agent == "reporter"
        jsonschema.validate(instance=final.to_dict(), schema=_load_schema("reporter"))
        assert final.payload["overall_severity"] in {"critical", "high", "medium", "low", "info"}

        # All leases were released on the way out.
        assert rt.residency.active_leases == 0
        store.close()


# ---------------------------------------------------------------------------
# (f) Every artifact in the EventLog passes schema validation
# ---------------------------------------------------------------------------


class TestEventLogArtifacts:
    async def test_all_logged_artifacts_validate(self, transport) -> None:
        rt = build_runtime(transport)
        case = make_case("case-log-validate")
        await rt.orchestrator.run_case(case)

        artifact_events = rt.orchestrator.event_log.events(case["case_id"], "artifact")
        # triage + 3 workers + reporter
        assert len(artifact_events) == 5
        for event in artifact_events:
            agent = event.data["agent"]
            raw = event.data["artifact_raw"]
            Artifact(raw, agent)  # re-validation must not raise
            jsonschema.validate(instance=raw, schema=_load_schema(agent))


# ---------------------------------------------------------------------------
# Intake + digest units
# ---------------------------------------------------------------------------


class TestBurstDetector:
    def test_below_threshold_no_case(self) -> None:
        records = _fixture_records()
        alerts = [r for r in records if r.level in {"WARN", "ERROR"}]
        detector = BurstDetector(threshold=len(alerts) + 1, window_s=3600.0)
        assert all(detector.observe(r) is None for r in sorted(records, key=lambda r: r.ts))

    def test_info_records_ignored(self) -> None:
        detector = BurstDetector(threshold=1, window_s=3600.0)
        infos = [r for r in _fixture_records() if r.level == "INFO"]
        assert all(detector.observe(r) is None for r in infos)

    def test_case_carries_trigger_records(self) -> None:
        detector = BurstDetector(threshold=3, window_s=60.0)
        case = None
        for record in sorted(_fixture_records(), key=lambda r: r.ts):
            case = detector.observe(record)
            if case:
                break
        assert case is not None
        assert case["record_count"] >= 3
        assert all(
            {"ts", "node", "level", "component", "message"} <= set(t)
            for t in case["trigger_records"]
        )


class TestDigestBuilder:
    def test_digest_summarises_store(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "digest.duckdb")
        store.write_records(_fixture_records())
        digest = DigestBuilder(store).build()
        assert "TELEMETRY DIGEST" in digest
        assert "records=" in digest
        assert "10.250.19.102" in digest
        assert "WARN" in digest
        store.close()

    def test_empty_store_digest(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "empty.duckdb")
        assert "records=0" in DigestBuilder(store).build()
        store.close()
