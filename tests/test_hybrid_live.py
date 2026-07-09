"""
Hybrid live-engine tests — Band B registry/generator/materializer, and
app/orchestrator/run_demo.py's --hybrid-live wiring + the dashboard's
engine_mode/hw_label honesty fields.

No real network anywhere: httpx.MockTransport stands in for both the
"real" engine and the in-process mock, so tests stay deterministic while
still exercising genuine httpx request construction (the actual JSON body
sent, not just internal state).
"""

from __future__ import annotations

import json

import httpx

from app.orchestrator.orchestrator import EventLog
from app.orchestrator.run_demo import HYBRID_LIVE_MODEL, _setup_contract_b
from core.budgeter import MemoryBudgeter
from core.generator import CoreGenerator
from core.materializer import LazyMaterializer
from core.memory_model import AGENT_MODEL_MAP, DEFAULT_MODELS, MemoryModel
from core.metrics import CoreMetrics
from core.registry import Engine, EngineRegistry, resolve_engine_transport
from core.residency import ResidencyManager
from dashboard.server import DashboardSources, build_state_snapshot, create_dashboard_app


def _recording_transport(served_models: set[str]) -> tuple[httpx.MockTransport, list[dict]]:
    """A fake Contract B engine: records every request's JSON body and
    answers /v1/models with `served_models`, /v1/chat/completions with a
    minimal valid completion echoing whatever model was requested."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": m} for m in served_models]}
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            calls.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}}],
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler), calls


def _unreachable_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated: real engine unreachable", request=request)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# core/registry.py — resolve_engine_transport sentinel
# ---------------------------------------------------------------------------


class TestResolveEngineTransport:
    def test_unset_falls_back_to_default(self) -> None:
        engine = Engine(model="m", base_url="http://x", port=8000)
        default = object()
        assert resolve_engine_transport(engine, default) is default

    def test_explicit_override_wins_even_when_none(self) -> None:
        # transport=None is a MEANINGFUL value (real network), not "unset".
        engine = Engine(model="m", base_url="http://x", port=8000, transport=None)
        default = object()
        assert resolve_engine_transport(engine, default) is None

    def test_explicit_transport_object_wins(self) -> None:
        override = object()
        engine = Engine(model="m", base_url="http://x", port=8000, transport=override)
        assert resolve_engine_transport(engine, object()) is override

    def test_served_model_defaults_to_none(self) -> None:
        engine = Engine(model="remanon-triage-7b", base_url="http://x", port=8000)
        assert engine.served_model is None


# ---------------------------------------------------------------------------
# core/generator.py + core/materializer.py — wire model name + per-engine
# transport, proven by inspecting the ACTUAL outgoing request body.
# ---------------------------------------------------------------------------


class TestWireModelRemapping:
    async def test_generate_sends_served_model_not_registry_key(self) -> None:
        transport, calls = _recording_transport({"gpt-oss-20b"})
        registry = EngineRegistry()
        registry.register(
            Engine(
                model="remanon-triage-7b",
                base_url="http://real-engine",
                port=8000,
                served_model="gpt-oss-20b",
                transport=transport,
            )
        )
        materializer = LazyMaterializer(registry, transport=transport)
        await materializer.materialize("ctx", "remanon-triage-7b")
        generator = CoreGenerator(registry, materializer, transport=transport)

        await generator.generate("triage", "remanon-triage-7b", "sys", "user")

        assert len(calls) == 2  # one prefill + one generate
        assert all(c["model"] == "gpt-oss-20b" for c in calls)  # never the placeholder name

    async def test_generate_sends_registry_key_when_served_model_unset(self) -> None:
        transport, calls = _recording_transport({"remanon-correlator-13b"})
        registry = EngineRegistry(transport=transport)
        registry.register(Engine(model="remanon-correlator-13b", base_url="http://mock", port=8000))
        materializer = LazyMaterializer(registry, transport=transport)
        await materializer.materialize("ctx", "remanon-correlator-13b")
        generator = CoreGenerator(registry, materializer, transport=transport)

        await generator.generate("correlator", "remanon-correlator-13b", "sys", "user")

        assert all(c["model"] == "remanon-correlator-13b" for c in calls)

    async def test_mixed_transport_registry_routes_each_engine_correctly(self) -> None:
        """The actual hybrid scenario: ONE registry, TWO engines, TWO
        different transports, in the SAME run — proves real+mock coexist."""
        real_transport, real_calls = _recording_transport({"gpt-oss-20b"})
        mock_transport, mock_calls = _recording_transport({"remanon-correlator-13b"})

        registry = EngineRegistry()  # no registry-wide default transport at all
        registry.register(
            Engine(
                model="remanon-triage-7b",
                base_url="http://real-engine",
                port=8000,
                served_model="gpt-oss-20b",
                transport=real_transport,
            )
        )
        registry.register(
            Engine(
                model="remanon-correlator-13b",
                base_url="http://mock-engine",
                port=8000,
                transport=mock_transport,
            )
        )
        materializer = LazyMaterializer(registry)
        generator = CoreGenerator(registry, materializer)

        await materializer.materialize("ctx", "remanon-triage-7b")
        await materializer.materialize("ctx", "remanon-correlator-13b")
        await generator.generate("triage", "remanon-triage-7b", "sys", "user")
        await generator.generate("correlator", "remanon-correlator-13b", "sys", "user")

        assert len(real_calls) == 2  # triage's prefill + generate, only
        assert all(c["model"] == "gpt-oss-20b" for c in real_calls)
        assert len(mock_calls) == 2  # correlator's prefill + generate, only
        assert all(c["model"] == "remanon-correlator-13b" for c in mock_calls)

    async def test_health_check_uses_served_model_for_the_served_check(self) -> None:
        transport, _ = _recording_transport({"gpt-oss-20b"})  # server only knows the real name
        registry = EngineRegistry()
        registry.register(
            Engine(
                model="remanon-triage-7b",  # registry key the server has NEVER heard of
                base_url="http://real-engine",
                port=8000,
                served_model="gpt-oss-20b",
                transport=transport,
            )
        )
        health = await registry.health_check()
        assert health["remanon-triage-7b"] is True

    async def test_health_check_false_when_served_model_not_offered(self) -> None:
        transport, _ = _recording_transport({"some-other-model"})
        registry = EngineRegistry()
        registry.register(
            Engine(
                model="remanon-triage-7b",
                base_url="http://real-engine",
                port=8000,
                served_model="gpt-oss-20b",
                transport=transport,
            )
        )
        health = await registry.health_check()
        assert health["remanon-triage-7b"] is False


# ---------------------------------------------------------------------------
# run_demo._setup_contract_b — the CLI-level decision logic
# ---------------------------------------------------------------------------


class TestSetupContractB:
    async def test_no_base_url_is_full_mock(self) -> None:
        registry, transport, engine_mode, hw_label = await _setup_contract_b(
            "", False, "AMD gfx1100 48GB"
        )
        assert engine_mode == "mock"
        assert hw_label is None
        for model in DEFAULT_MODELS:
            assert registry.resolve(model).served_model is None

    async def test_hybrid_live_healthy_splits_triage_real_rest_mock(self) -> None:
        real_transport, real_calls = _recording_transport({HYBRID_LIVE_MODEL})
        registry, transport, engine_mode, hw_label = await _setup_contract_b(
            "http://localhost:8000", True, "AMD gfx1100 48GB", real_transport=real_transport
        )

        assert engine_mode == "live"
        assert hw_label == "AMD gfx1100 48GB"

        triage_model = AGENT_MODEL_MAP["triage"]
        triage_engine = registry.resolve(triage_model)
        assert triage_engine.served_model == HYBRID_LIVE_MODEL
        assert triage_engine.base_url == "http://localhost:8000"
        assert resolve_engine_transport(triage_engine, transport) is real_transport

        # Every other distinct model stays on the in-process mock, not real_transport.
        for model in DEFAULT_MODELS:
            if model == triage_model:
                continue
            other = registry.resolve(model)
            assert other.served_model is None
            assert resolve_engine_transport(other, transport) is not real_transport

        # And the split is REAL at the wire level: materializing triage hits
        # the real transport with the real served name; nothing else does.
        materializer = LazyMaterializer(registry, transport=transport)
        await materializer.materialize("ctx", triage_model)
        assert len(real_calls) == 1
        assert real_calls[0]["model"] == HYBRID_LIVE_MODEL

    async def test_hybrid_live_unreachable_falls_back_to_full_mock(self, capsys) -> None:
        registry, transport, engine_mode, hw_label = await _setup_contract_b(
            "http://localhost:8000",
            True,
            "AMD gfx1100 48GB",
            real_transport=_unreachable_transport(),
        )

        assert engine_mode == "mock"
        assert hw_label is None
        # Never crashed — a clear warning was printed instead.
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "falling back to full in-process mock" in captured.err

        # And the fallback registry actually works end-to-end (never a
        # half-wired, silently-broken state).
        triage_model = AGENT_MODEL_MAP["triage"]
        materializer = LazyMaterializer(registry, transport=transport)
        handle = await materializer.materialize("ctx", triage_model)
        assert handle is not None

    async def test_plain_base_url_without_hybrid_is_fully_live(self) -> None:
        real_transport, real_calls = _recording_transport(set(DEFAULT_MODELS))
        registry, transport, engine_mode, hw_label = await _setup_contract_b(
            "http://localhost:8000", False, "AMD gfx1100 48GB", real_transport=real_transport
        )

        assert engine_mode == "live"
        assert hw_label == "AMD gfx1100 48GB"
        for model in DEFAULT_MODELS:
            assert registry.resolve(model).base_url == "http://localhost:8000"


# ---------------------------------------------------------------------------
# dashboard/server.py — engine_mode/hw_label drive the honest LIVE badge
# ---------------------------------------------------------------------------


def _sources(engine_mode: str = "mock", hw_label: str | None = None) -> DashboardSources:
    mm = MemoryModel()
    residency = ResidencyManager()
    budgeter = MemoryBudgeter(mm)
    registry = EngineRegistry()
    materializer = LazyMaterializer(registry)
    metrics = CoreMetrics(
        memory_model=mm, materializer=materializer, residency=residency, budgeter=budgeter
    )
    return DashboardSources(
        event_log=EventLog(),
        metrics=metrics,
        memory_model=mm,
        residency=residency,
        engine_mode=engine_mode,
        hw_label=hw_label,
    )


class TestDashboardLiveBadge:
    def test_mock_mode_reads_replay(self) -> None:
        sources = _sources("mock", None)
        assert sources.live_badge == "REPLAY (recorded real run)"

    def test_live_mode_reads_live_with_hw_label(self) -> None:
        sources = _sources("live", "AMD gfx1100 48GB")
        assert sources.live_badge == "LIVE — AMD gfx1100 48GB"

    def test_live_mode_without_hw_label_falls_back_honestly(self) -> None:
        # engine_mode="live" with no label is a contradiction the run_demo
        # wiring never produces, but the badge must still never lie by
        # printing a bare "LIVE" with no hardware attached.
        sources = _sources("live", None)
        assert sources.live_badge == "REPLAY (recorded real run)"

    def test_api_state_exposes_engine_mode_hw_label_and_live_badge(self) -> None:
        sources = _sources("live", "AMD gfx1100 48GB")
        state = build_state_snapshot(sources)
        assert state["engine_mode"] == "live"
        assert state["hw_label"] == "AMD gfx1100 48GB"
        assert state["live_badge"] == "LIVE — AMD gfx1100 48GB"

    def test_engine_mode_flips_to_live_when_a_real_engine_is_registered(self) -> None:
        """Direct assertion the task asked for: engine_mode reflects
        whether a real engine is actually wired in, not a fixed default."""
        mock_sources = _sources("mock", None)
        live_sources = _sources("live", "AMD gfx1100 48GB")
        assert mock_sources.engine_mode == "mock"
        assert live_sources.engine_mode == "live"
        assert mock_sources.live_badge != live_sources.live_badge

    async def test_dashboard_app_serves_live_badge_over_http(self) -> None:
        sources = _sources("live", "AMD gfx1100 48GB")
        app = create_dashboard_app(sources)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://dash"
        ) as client:
            response = await client.get("/api/state")
        assert response.status_code == 200
        body = response.json()
        assert body["live_badge"] == "LIVE — AMD gfx1100 48GB"
