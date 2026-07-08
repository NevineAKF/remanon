"""
Layer L9 dashboard tests — no browser, no network at all.

HTTP is exercised through httpx ASGITransport; the WebSocket ordering test
drives the ASGI websocket interface directly in the test's own event loop
(no server, no sockets, fully deterministic). The route-table test proves
the observation plane is read-only by construction.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import httpx
import pytest

from app.orchestrator.orchestrator import EventLog
from contracts.contract_a import Artifact
from core.budgeter import MemoryBudgeter
from core.memory_model import DEFAULT_MODELS, MemoryModel
from core.metrics import CoreMetrics
from dashboard.server import DashboardSources, create_dashboard_app
from deploy.mock_engine.main import app as mock_app
from tests.test_orchestrator import Runtime, build_runtime, make_case

_FORBIDDEN_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


@pytest.fixture()
def transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=mock_app)


def build_dashboard(transport: httpx.ASGITransport) -> tuple[Runtime, MemoryModel, object]:
    rt = build_runtime(transport)
    memory_model = MemoryModel()
    metrics = CoreMetrics(
        memory_model=memory_model,
        materializer=rt.materializer,
        residency=rt.residency,
        budgeter=MemoryBudgeter(memory_model),
    )
    dash_app = create_dashboard_app(
        DashboardSources(
            event_log=rt.orchestrator.event_log,
            metrics=metrics,
            memory_model=memory_model,
            residency=rt.residency,
        )
    )
    return rt, memory_model, dash_app


async def get_state(dash_app) -> dict:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=dash_app), base_url="http://dash"
    ) as client:
        response = await client.get("/api/state")
    assert response.status_code == 200
    return response.json()


# ---------------------------------------------------------------------------
# /api/state schema + exact ledger arithmetic
# ---------------------------------------------------------------------------


class TestApiState:
    async def test_schema_and_ledger_sums(self, transport) -> None:
        rt, memory_model, dash_app = build_dashboard(transport)
        budgeter = MemoryBudgeter(memory_model)
        # Rebuild metrics over the pinned budgeter so the ledger is non-trivial.
        metrics = CoreMetrics(
            memory_model=memory_model,
            materializer=rt.materializer,
            residency=rt.residency,
            budgeter=budgeter,
        )
        for model in DEFAULT_MODELS:
            await budgeter.pin_model(model)
        dash_app = create_dashboard_app(
            DashboardSources(
                event_log=rt.orchestrator.event_log,
                metrics=metrics,
                memory_model=memory_model,
                residency=rt.residency,
            )
        )

        state = await get_state(dash_app)

        for key in (
            "metrics",
            "capacity_gb",
            "headroom_gb",
            "masters_config",
            "active_leases",
            "cases_processed",
            "events",
            "verdicts",
        ):
            assert key in state, f"missing key: {key}"

        ledger = state["metrics"]["ledger"]
        # The memory numbers must sum EXACTLY (byte-exact ledger underneath).
        assert (
            ledger["weights_gb"] + ledger["masters_gb"] + ledger["deltas_gb"] == ledger["used_gb"]
        )
        assert ledger["used_gb"] + ledger["available_gb"] == ledger["usable_gb"]
        assert ledger["usable_gb"] == state["capacity_gb"] - state["headroom_gb"]
        assert ledger["weights_gb"] == 80.0
        assert ledger["masters_gb"] == 32.0

        assert len(state["masters_config"]) == len(DEFAULT_MODELS)
        assert all(entry["master_gb"] > 0 for entry in state["masters_config"])

    async def test_index_page_served(self, transport) -> None:
        _, _, dash_app = build_dashboard(transport)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=dash_app), base_url="http://dash"
        ) as client:
            response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "REMANON" in response.text
        assert "/ws/events" in response.text  # the page wires the live socket


# ---------------------------------------------------------------------------
# Read-only by construction
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_route_table_has_zero_mutating_routes(self) -> None:
        transport = httpx.ASGITransport(app=mock_app)
        _, _, dash_app = build_dashboard(transport)

        checked = 0
        for route in dash_app.routes:
            methods = getattr(route, "methods", None)
            if methods is None:
                continue  # WebSocket route — no HTTP methods by definition
            assert not (set(methods) & _FORBIDDEN_METHODS), (
                f"mutating method on {route.path}: {methods}"
            )
            checked += 1
        assert checked >= 2  # at least "/" and "/api/state" were inspected


# ---------------------------------------------------------------------------
# Event streaming
# ---------------------------------------------------------------------------


class TestEventStreaming:
    async def test_eventlog_subscription_delivers_in_order(self) -> None:
        log = EventLog()
        queue = log.subscribe()
        for i in range(5):
            log.append("case-sub", "state", state=f"S{i}")
        received = [await asyncio.wait_for(queue.get(), timeout=1.0) for _ in range(5)]
        assert [e.data["state"] for e in received] == [f"S{i}" for i in range(5)]

        log.unsubscribe(queue)
        log.append("case-sub", "state", state="after-unsubscribe")
        assert queue.empty()

    async def test_websocket_streams_appended_events_in_order(self, transport) -> None:
        """Drives the /ws/events ASGI endpoint directly — no sockets, one event loop."""
        rt, _, dash_app = build_dashboard(transport)

        outbound: asyncio.Queue = asyncio.Queue()  # messages the app sends to the client
        inbound: asyncio.Queue = asyncio.Queue()  # messages the client sends to the app
        await inbound.put({"type": "websocket.connect"})

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "scheme": "ws",
            "path": "/ws/events",
            "raw_path": b"/ws/events",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "subprotocols": [],
            "client": ("testclient", 0),
            "server": ("testserver", 80),
        }

        connection_task = asyncio.create_task(dash_app(scope, inbound.get, outbound.put))
        try:
            accept = await asyncio.wait_for(outbound.get(), timeout=2.0)
            assert accept["type"] == "websocket.accept"

            log = rt.orchestrator.event_log
            log.append("case-ws", "state", state="ONE")
            log.append("case-ws", "worker_start", agent="hunter")
            log.append("case-ws", "state", state="TWO")

            received = []
            for _ in range(3):
                message = await asyncio.wait_for(outbound.get(), timeout=2.0)
                assert message["type"] == "websocket.send"
                received.append(json.loads(message["text"]))

            assert [r["kind"] for r in received] == ["state", "worker_start", "state"]
            assert received[0]["data"]["state"] == "ONE"
            assert received[1]["data"]["agent"] == "hunter"
            assert received[2]["data"]["state"] == "TWO"
            assert all(r["case_id"] == "case-ws" for r in received)
        finally:
            connection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await connection_task

        # Cancellation ran the handler's finally: the subscriber is gone, so
        # further appends must not accumulate anywhere.
        rt.orchestrator.event_log.append("case-ws", "state", state="AFTER")
        assert outbound.empty()


# ---------------------------------------------------------------------------
# End-to-end: run a case, verdict lands in /api/state
# ---------------------------------------------------------------------------


class TestEndToEndDashboard:
    async def test_verdict_appears_in_state_after_case(self, transport) -> None:
        rt, _, dash_app = build_dashboard(transport)

        result = await rt.orchestrator.run_case(make_case("case-dash-e2e"))
        assert isinstance(result, Artifact)

        state = await get_state(dash_app)

        assert state["cases_processed"] == 1
        assert len(state["verdicts"]) == 1
        verdict = state["verdicts"][0]
        assert verdict["case_id"] == "case-dash-e2e"
        assert verdict["title"] == result.payload["title"]
        assert verdict["overall_severity"] == result.payload["overall_severity"]

        assert len(state["events"]) > 0
        state_events = [e for e in state["events"] if e["kind"] == "state"]
        assert state_events[-1]["data"]["state"] == "EMIT"
        # All leases released → live lease view is empty.
        assert state["active_leases"] == []
        assert state["metrics"]["gb_saved_vs_per_agent"] > 0


# ---------------------------------------------------------------------------
# Incident report export — read-only, generated from the same EventLog
# ---------------------------------------------------------------------------


class TestReportEndpoints:
    async def test_report_csv_endpoint(self, transport) -> None:
        rt, _, dash_app = build_dashboard(transport)
        await rt.orchestrator.run_case(make_case("case-report-csv"))

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=dash_app), base_url="http://dash"
        ) as client:
            response = await client.get("/api/report.csv")

        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "case-report-csv" in response.text
        assert response.text.splitlines()[0].startswith("case_id,")

    async def test_report_md_endpoint(self, transport) -> None:
        rt, _, dash_app = build_dashboard(transport)
        await rt.orchestrator.run_case(make_case("case-report-md"))

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=dash_app), base_url="http://dash"
        ) as client:
            response = await client.get("/api/report.md")

        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
        assert response.text.startswith("# REMANON")
        assert "case-rep" in response.text  # case_id[:8] appears in a section heading

    async def test_report_endpoints_do_not_mutate_state(self, transport) -> None:
        """Calling both report endpoints must not change /api/state's view."""
        rt, _, dash_app = build_dashboard(transport)
        await rt.orchestrator.run_case(make_case("case-report-readonly"))
        before = await get_state(dash_app)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=dash_app), base_url="http://dash"
        ) as client:
            await client.get("/api/report.csv")
            await client.get("/api/report.md")

        after = await get_state(dash_app)
        assert before["events"] == after["events"]
        assert before["verdicts"] == after["verdicts"]
        assert before["metrics"] == after["metrics"]
