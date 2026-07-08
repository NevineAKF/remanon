"""
Band A, Layer L9 — dashboard server (read-only observation plane).

Serves the static page, GET /api/state (JSON snapshot), and WebSocket
/ws/events streaming EventLog entries live. STRICTLY READ-ONLY by
construction: only GET routes and one WebSocket exist — the observation
plane cannot mutate Band B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from app.config.evidence import load_measured_evidence
from app.config.hardware import HardwareProfile, active_profile
from app.orchestrator.orchestrator import EventLog
from app.orchestrator.report import (
    build_incident_report,
    extract_cases_from_events,
    render_csv_report,
    render_markdown_report,
)
from core.memory_model import MemoryModel
from core.metrics import CoreMetrics
from core.residency import ResidencyManager

_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@dataclass
class DashboardSources:
    """Read-only references into the running system."""

    event_log: EventLog
    metrics: CoreMetrics
    memory_model: MemoryModel
    residency: ResidencyManager
    # The ACTIVE hardware this run is on — env-selected (REMANON_HW), never
    # hardcoded to the unmeasured MI300X. Independent of memory_model's
    # capacity_gb, which is Core's own fixed COMPUTED placeholder — the two
    # deliberately do not have to agree; each is labeled honestly instead.
    hardware_profile: HardwareProfile = field(default_factory=active_profile)
    memory_tech: str = "HBM3"
    # "mock" until D-03 hardware validation; flips to "live" on real silicon.
    engine_mode: str = "mock"

    @property
    def hardware_name(self) -> str:
        return self.hardware_profile.name


def build_state_snapshot(sources: DashboardSources, last_events: int = 200) -> dict[str, Any]:
    """
    The exact payload GET /api/state serves — factored out so the showcase
    recorder (dashboard/recorder.py) can capture the same shape offline,
    in-process, without an HTTP round trip.
    """
    mm = sources.memory_model
    profile = sources.hardware_profile
    evidence = load_measured_evidence()
    events = sources.event_log.events()

    verdicts = []
    for event in events:
        if event.kind == "artifact" and event.data.get("agent") == "reporter":
            payload = event.data["artifact_raw"]["payload"]
            verdicts.append(
                {
                    "case_id": event.case_id,
                    "ts": event.ts.isoformat(),
                    "title": payload["title"],
                    "overall_severity": payload["overall_severity"],
                    "executive_summary": payload["executive_summary"],
                }
            )

    return {
        "metrics": sources.metrics.export(),
        "capacity_gb": mm.total_capacity_gb,
        "capacity_source": "computed",  # MI300X 192 GB target — core/memory_model.py, not measured
        "headroom_gb": mm.headroom_gb,
        "headroom_source": "computed",
        # Honest hardware label: the ACTIVE profile (env REMANON_HW, default
        # MEASURED) — independent of capacity_gb above. Never hardcodes MI300X.
        "hardware": profile.label,
        "hardware_profile": {
            "name": profile.name,
            "gfx": profile.gfx,
            "vram_gb": profile.vram_gb,
            "source": profile.source,
            "label": profile.label,
        },
        "memory_tech": sources.memory_tech,
        "engine_mode": sources.engine_mode,
        "masters_config": [
            {
                "model": spec.name,
                "weights_gb": spec.weights_gb,
                "master_gb": spec.master_gb,
                "source": "computed",  # sized for the 192 GB MI300X target, not measured
            }
            for spec in mm.models.values()
        ],
        # The six real, directly-measured numbers (docs/evidence/) — always
        # exposed as-is, never overwritten by a synthetic/placeholder value.
        "evidence": evidence.to_dict(),
        "active_leases": [
            {
                "lease_id": lease.lease_id,
                "agent": lease.agent,
                "model": lease.model,
                "created_at": lease.created_at.isoformat(),
            }
            for lease in sources.residency.lease_table.active()
        ],
        "cases_processed": sum(
            1 for e in events if e.kind == "state" and e.data.get("state") == "EMIT"
        ),
        "events": [e.to_dict() for e in events[-last_events:]],
        "verdicts": verdicts,
    }


def create_dashboard_app(sources: DashboardSources, last_events: int = 200) -> FastAPI:
    app = FastAPI(title="Remanon Dashboard (L9)", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_INDEX_HTML)

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return build_state_snapshot(sources, last_events)

    def _build_report() -> dict[str, Any]:
        events = sources.event_log.events()
        ledger = sources.metrics.export()["ledger"]
        cases = extract_cases_from_events(events, ledger)
        return build_incident_report(
            cases, sources.metrics.export(), engine_mode=sources.engine_mode
        )

    @app.get("/api/report.csv")
    async def report_csv() -> PlainTextResponse:
        return PlainTextResponse(
            render_csv_report(_build_report()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=incident_report.csv"},
        )

    @app.get("/api/report.md")
    async def report_md() -> PlainTextResponse:
        return PlainTextResponse(
            render_markdown_report(_build_report()),
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=incident_report.md"},
        )

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = sources.event_log.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            sources.event_log.unsubscribe(queue)

    return app
