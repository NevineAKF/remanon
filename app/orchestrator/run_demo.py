"""
Band A, Layer L8 — end-to-end demo CLI.

Boots the core (materialize + pin masters via the mock engine), streams the
telemetry store through the Replayer, opens cases via the burst rule, runs
the full pipeline, and prints each verdict plus a final metrics snapshot.

Usage:
    python -m app.orchestrator.run_demo --log-file tests/fixtures/hdfs_sample.log --speed 1000
    python -m app.orchestrator.run_demo --db data/store/telemetry.duckdb --speed 3600
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import typer
import uvicorn

from app.adapter.digest import DigestBuilder
from app.agents.base import MASTER_CONTEXT_ID, BaseAgent
from app.agents.correlator_agent import CorrelatorAgent
from app.agents.hunter_agent import HunterAgent
from app.agents.reporter_agent import ReporterAgent
from app.agents.topology_agent import TopologyAgent
from app.agents.triage_agent import TriageAgent
from app.dataplane.normalizer import HDFSNormalizer
from app.dataplane.parser import parse_file
from app.dataplane.replayer import stream
from app.dataplane.store import TelemetryStore
from app.orchestrator.orchestrator import BurstDetector, EventLog, Orchestrator
from contracts.contract_a import Artifact
from core.budgeter import MemoryBudgeter
from core.generator import CoreGenerator
from core.materializer import LazyMaterializer
from core.memory_model import DEFAULT_MODELS, MemoryModel
from core.metrics import CoreMetrics
from core.registry import EngineRegistry, default_engines
from core.residency import ResidencyManager
from dashboard.server import DashboardSources, create_dashboard_app
from deploy.mock_engine.main import app as mock_app

cli = typer.Typer(add_completion=False)

_LOG_FILE_OPTION = typer.Option(None, help="Raw HDFS log to load instead of the DuckDB store.")
_DB_OPTION = typer.Option(Path("data/store/telemetry.duckdb"), help="TelemetryStore DuckDB path.")


@cli.command()
def demo(
    log_file: Path = _LOG_FILE_OPTION,
    db: Path = _DB_OPTION,
    speed: float = typer.Option(60.0, help="Replay speed factor (log-seconds per wall-second)."),
    # Intake defaults are measured, not guessed: the HDFS_2k store profile shows
    # a maximum of 5 alerts in any 300 s window (at 2008-11-10 08:30:45), while
    # 60 s windows never exceed 3 — hence n=4 within 300 s.
    burst_n: int = typer.Option(
        4, "--burst-n", help="WARN/ERROR count that opens a case (default from HDFS_2k profile)."
    ),
    burst_window_s: float = typer.Option(
        300.0,
        "--burst-window-s",
        help="Sliding window in log-seconds, over ORIGINAL record timestamps "
        "(independent of --speed).",
    ),
    base_url: str = typer.Option(
        "", help="Live Contract B engine URL; empty = in-process mock engine."
    ),
    max_cases: int = typer.Option(5, help="Stop after this many cases."),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Serve the L9 observation plane during replay."
    ),
    dashboard_port: int = typer.Option(8080, help="Dashboard port."),
) -> None:
    asyncio.run(
        _run(
            log_file,
            db,
            speed,
            burst_n,
            burst_window_s,
            base_url,
            max_cases,
            dashboard,
            dashboard_port,
        )
    )


async def _run(
    log_file: Path | None,
    db: Path,
    speed: float,
    burst_n: int,
    burst_window_s: float,
    base_url: str,
    max_cases: int,
    dashboard: bool,
    dashboard_port: int,
) -> None:
    # --- telemetry store ---
    if log_file is not None:
        store = TelemetryStore(Path(":memory:"))
        parsed = parse_file(log_file)
        normalizer = HDFSNormalizer()
        store.write_records([normalizer.normalize(p) for p in parsed.records])
        typer.echo(f"Loaded {store.count()} records from {log_file} (skipped {parsed.skipped})")
    else:
        store = TelemetryStore(db)
        if store.count() == 0:
            typer.echo(
                "Store is empty — run `python -m app.dataplane.ingest --dataset hdfs_2k` first, "
                "or pass --log-file.",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Using store {db} with {store.count()} records")

    # --- burst profile: measured intake feasibility, printed up front ---
    profile = store.burst_profile(burst_window_s)
    if profile is None:
        typer.echo(f"burst profile: no WARN/ERROR records in store (window={burst_window_s:g}s)")
    else:
        max_n, at = profile
        typer.echo(
            f"burst profile: max {max_n} alerts in any {burst_window_s:g}s window "
            f"(at {at.isoformat()}); intake rule: >= {burst_n} alerts in {burst_window_s:g}s"
        )
        if burst_n > max_n:
            typer.echo(
                f"WARNING: --burst-n={burst_n} exceeds the dataset maximum ({max_n}) — "
                "this run will open ZERO cases.",
                err=True,
            )

    # --- Contract B endpoint ---
    if base_url:
        transport: httpx.AsyncBaseTransport | None = None
        engine_url = base_url
    else:
        transport = httpx.ASGITransport(app=mock_app)
        engine_url = "http://mock-engine"
        typer.echo("Contract B: in-process mock engine")

    # --- boot the core ---
    memory_model = MemoryModel()
    masters = ", ".join(
        f"{spec.name}={spec.master_gb:g}GB" for spec in memory_model.models.values()
    )
    typer.echo(f"memory model masters (placeholders pending D-03): {masters}")
    budgeter = MemoryBudgeter(memory_model)
    residency = ResidencyManager(on_release=lambda lease: budgeter.release_delta(lease.lease_id))
    registry = EngineRegistry(transport=transport)
    for engine in default_engines(engine_url):
        registry.register(engine)
    materializer = LazyMaterializer(registry, transport=transport)
    generator = CoreGenerator(registry, materializer, transport=transport)
    metrics = CoreMetrics(
        memory_model=memory_model,
        materializer=materializer,
        residency=residency,
        budgeter=budgeter,
    )

    digest = DigestBuilder(store).build()
    typer.echo("--- master digest ---")
    typer.echo(digest)
    for model in DEFAULT_MODELS:
        await budgeter.pin_model(model)
        await materializer.materialize(MASTER_CONTEXT_ID, model, context_text=digest)
    typer.echo(f"Pinned + materialized {len(DEFAULT_MODELS)} masters")

    # --- agents + orchestrator ---
    deps: dict = {"residency": residency, "materializer": materializer, "generator": generator}
    agents: dict[str, BaseAgent] = {
        "triage": TriageAgent(**deps),
        "correlator": CorrelatorAgent(**deps),
        "hunter": HunterAgent(store=store, **deps),
        "topology": TopologyAgent(**deps),
        "reporter": ReporterAgent(**deps),
    }
    orchestrator = Orchestrator(agents, EventLog())
    detector = BurstDetector(threshold=burst_n, window_s=burst_window_s)

    # --- optional L9 dashboard (read-only observation plane) ---
    dashboard_task: asyncio.Task | None = None
    if dashboard:
        dash_app = create_dashboard_app(
            DashboardSources(
                event_log=orchestrator.event_log,
                metrics=metrics,
                memory_model=memory_model,
                residency=residency,
            )
        )
        dash_config = uvicorn.Config(
            dash_app, host="127.0.0.1", port=dashboard_port, log_level="warning"
        )
        dash_server = uvicorn.Server(dash_config)
        dashboard_task = asyncio.create_task(dash_server.serve())
        typer.echo(f"Dashboard live at http://127.0.0.1:{dashboard_port} (read-only)")

    # --- replay + pipeline ---
    cases_processed = 0
    typer.echo(f"--- replaying at {speed}x ---")
    event_log = orchestrator.event_log
    async for record in stream(store.all_records(), speed_factor=speed):
        # Feed the observation plane: one "record" event per replayed record,
        # so every particle the L9 theater draws is a real Loghub line.
        event_log.append(
            "replay",
            "record",
            ts=record.ts.isoformat(),
            node=record.node,
            level=record.level,
            component=record.component,
            message=record.message[:160],
            dialect=record.dialect,
        )
        case = detector.observe(record)
        if case is None:
            continue
        event_log.append(
            case["case_id"],
            "case_open",
            opened_at=case["opened_at"],
            record_count=case["record_count"],
            trigger_records=case["trigger_records"],
        )
        nodes = ",".join(sorted({t["node"] for t in case["trigger_records"]}))
        typer.echo(
            f"\n[case-open] t={case['opened_at']} triggers={case['record_count']} "
            f"nodes={nodes} id={case['case_id'][:8]}"
        )
        verdict = await orchestrator.run_case(case)
        cases_processed += 1
        _print_verdict(verdict)
        if cases_processed >= max_cases:
            typer.echo(f"Reached --max-cases={max_cases}, stopping replay.")
            break

    if cases_processed == 0:
        typer.echo(
            "\nNo cases opened. Check the burst profile above and lower --burst-n "
            "or widen --burst-window-s."
        )

    # --- final metrics snapshot ---
    snapshot = metrics.export()
    typer.echo("\n--- metrics snapshot ---")
    typer.echo(f"cases_processed      : {cases_processed}")
    typer.echo(f"prefills_performed   : {snapshot['prefills_performed']}")
    typer.echo(f"prefills_avoided     : {snapshot['prefills_avoided']}")
    typer.echo(f"gb_saved_vs_per_agent: {snapshot['gb_saved_vs_per_agent']}")
    typer.echo(f"active_leases        : {snapshot['active_leases']}")
    typer.echo(f"ledger.used_gb       : {snapshot['ledger']['used_gb']}")
    typer.echo(f"event_log entries    : {len(orchestrator.event_log)}")

    if dashboard_task is not None:
        typer.echo(
            f"\nDashboard still serving at http://127.0.0.1:{dashboard_port} — Ctrl+C to exit."
        )
        await dashboard_task

    store.close()


def _print_verdict(verdict: Artifact | dict) -> None:
    if isinstance(verdict, Artifact):
        payload = verdict.payload
        typer.echo(f"VERDICT [{payload['overall_severity'].upper()}] {payload['title']}")
        typer.echo(f"  {payload['executive_summary']}")
        for section in payload["sections"]:
            typer.echo(f"  - {section['heading']}: {section['body']}")
    else:
        typer.echo(f"VERDICT (degraded): {verdict}")


if __name__ == "__main__":
    cli()
