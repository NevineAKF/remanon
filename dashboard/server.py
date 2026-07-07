"""
Band A, Layer L9 — dashboard server (read-only observation plane).

Serves the static page, GET /api/state (JSON snapshot), and WebSocket
/ws/events streaming EventLog entries live. STRICTLY READ-ONLY by
construction: only GET routes and one WebSocket exist — the observation
plane cannot mutate Band B.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.orchestrator.orchestrator import EventLog
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
    # Deployment labels for the observation plane; capacity itself always
    # comes from the MemoryModel so the two can never disagree.
    hardware_name: str = "AMD Instinct™ MI300X"
    memory_tech: str = "HBM3"


def create_dashboard_app(sources: DashboardSources, last_events: int = 200) -> FastAPI:
    app = FastAPI(title="Remanon Dashboard (L9)", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_INDEX_HTML)

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        mm = sources.memory_model
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
            "headroom_gb": mm.headroom_gb,
            "hardware": (
                f"{sources.hardware_name} · {mm.total_capacity_gb:g} GB {sources.memory_tech}"
            ),
            "memory_tech": sources.memory_tech,
            "masters_config": [
                {"model": spec.name, "weights_gb": spec.weights_gb, "master_gb": spec.master_gb}
                for spec in mm.models.values()
            ],
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
