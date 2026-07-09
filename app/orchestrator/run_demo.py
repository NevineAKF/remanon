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
import time
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
from app.orchestrator.report import (
    build_incident_report,
    export_incident_report,
    extract_cases_from_events,
)
from contracts.contract_a import Artifact
from core.budgeter import MemoryBudgeter
from core.generator import CoreGenerator
from core.materializer import LazyMaterializer
from core.memory_model import AGENT_MODEL_MAP, DEFAULT_MODELS, MemoryModel
from core.metrics import CoreMetrics
from core.registry import Engine, EngineRegistry, default_engines
from core.residency import ResidencyManager
from dashboard.recorder import RunRecorder, embed_recording_in_showcase, write_recording
from dashboard.server import DashboardSources, create_dashboard_app
from deploy.mock_engine.main import app as mock_app

cli = typer.Typer(add_completion=False)

_LOG_FILE_OPTION = typer.Option(None, help="Raw HDFS log to load instead of the DuckDB store.")
_DB_OPTION = typer.Option(Path("data/store/telemetry.duckdb"), help="TelemetryStore DuckDB path.")

# The real checkpoint name Triage's internal placeholder model
# ("remanon-triage-7b", per AGENT_MODEL_MAP) stands in for — see
# docs/evidence/D03_budget_sheet.md Tier 2. This is the only model name a
# --hybrid-live real engine is ever asked to serve.
HYBRID_LIVE_MODEL = "gpt-oss-20b"


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
    hybrid_live: bool = typer.Option(
        False,
        "--hybrid-live",
        help=f"With --base-url set: send ONLY the Triage agent ({HYBRID_LIVE_MODEL}) to the "
        "real engine; every other agent stays on the in-process mock. Health-checks the "
        "real engine first and falls back to full mock (never crashes) if it's unreachable.",
    ),
    hw_label: str = typer.Option(
        "AMD gfx1100 48GB",
        "--hw-label",
        help="Hardware label the dashboard's LIVE badge shows when a real engine is wired.",
    ),
    max_cases: int = typer.Option(5, help="Stop after this many cases."),
    dashboard: bool = typer.Option(
        False, "--dashboard", help="Serve the L9 observation plane during replay."
    ),
    dashboard_port: int = typer.Option(8080, help="Dashboard port."),
    export: bool = typer.Option(
        True,
        "--export/--no-export",
        help="Write the incident report (CSV + Markdown) to reports/ after the run.",
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="Capture the complete run (EventLog + timestamped state snapshots) into "
        "dashboard/showcase/run_recording.json and embed it into the showcase page, "
        "for self-contained static hosting (GitHub Pages).",
    ),
) -> None:
    asyncio.run(
        _run(
            log_file,
            db,
            speed,
            burst_n,
            burst_window_s,
            base_url,
            hybrid_live,
            hw_label,
            max_cases,
            dashboard,
            dashboard_port,
            export,
            record,
        )
    )


async def _setup_contract_b(
    base_url: str,
    hybrid_live: bool,
    hw_label: str,
    *,
    mock_app_: object = mock_app,
    real_transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[EngineRegistry, httpx.AsyncBaseTransport | None, str, str | None]:
    """
    Wire Contract B for this run. Returns (registry, default_transport,
    engine_mode, active_hw_label).

    - No base_url: full in-process mock. engine_mode="mock".
    - base_url + --hybrid-live: health-checks the real engine FIRST (never
      claims "live" for an engine that never actually answered). If it
      answers for HYBRID_LIVE_MODEL, Triage's engine goes real (with its
      served_model remapped); every other model stays on the in-process
      mock, in the SAME registry, at the SAME time. engine_mode="live". If
      the health check fails, falls back to full mock with a printed
      warning — never crashes the demo.
    - base_url alone (no --hybrid-live): the pre-existing fully-real path,
      unchanged, now correctly labeled engine_mode="live".

    `real_transport` exists only for tests, so they can stand in a fake
    "real" engine without touching the network — production code never
    passes it (None = genuine real TCP).
    """
    mock_transport = httpx.ASGITransport(app=mock_app_)
    triage_model = AGENT_MODEL_MAP["triage"]

    if not base_url:
        registry = EngineRegistry(transport=mock_transport)
        for engine in default_engines("http://mock-engine"):
            registry.register(engine)
        typer.echo("Contract B: in-process mock engine")
        return registry, mock_transport, "mock", None

    if hybrid_live:
        probe = EngineRegistry()
        probe.register(
            Engine(
                model=triage_model,
                base_url=base_url,
                port=8000,
                served_model=HYBRID_LIVE_MODEL,
                transport=real_transport,
            )
        )
        healthy = (await probe.health_check()).get(triage_model, False)
        if healthy:
            registry = EngineRegistry()
            registry.register(
                Engine(
                    model=triage_model,
                    base_url=base_url,
                    port=8000,
                    served_model=HYBRID_LIVE_MODEL,
                    transport=real_transport,
                )
            )
            for model in DEFAULT_MODELS:
                if model == triage_model:
                    continue
                registry.register(
                    Engine(
                        model=model,
                        base_url="http://mock-engine",
                        port=8000,
                        transport=mock_transport,
                    )
                )
            typer.echo(
                f"Contract B: HYBRID LIVE — Triage ({HYBRID_LIVE_MODEL}) -> {base_url} "
                "(real, verified reachable); correlator/hunter/topology -> in-process mock"
            )
            return registry, None, "live", hw_label

        typer.echo(
            f"WARNING: --hybrid-live requested but the real engine at {base_url} did not "
            f"respond healthy for {HYBRID_LIVE_MODEL!r} — falling back to full in-process "
            "mock.",
            err=True,
        )
        registry = EngineRegistry(transport=mock_transport)
        for engine in default_engines("http://mock-engine"):
            registry.register(engine)
        return registry, mock_transport, "mock", None

    # Plain --base-url, no --hybrid-live: pre-existing fully-real path.
    registry = EngineRegistry(transport=real_transport)
    for engine in default_engines(base_url):
        registry.register(engine)
    typer.echo(f"Contract B: fully live at {base_url}")
    return registry, real_transport, "live", hw_label


async def _run(
    log_file: Path | None,
    db: Path,
    speed: float,
    burst_n: int,
    burst_window_s: float,
    base_url: str,
    hybrid_live: bool,
    hw_label: str,
    max_cases: int,
    dashboard: bool,
    dashboard_port: int,
    export: bool,
    record: bool,
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

    # --- Contract B endpoint(s) — mock, fully live, or hybrid live ---
    registry, transport, engine_mode, hw_label_active = await _setup_contract_b(
        base_url, hybrid_live, hw_label
    )

    # --- boot the core ---
    memory_model = MemoryModel()
    masters = ", ".join(
        f"{spec.name}={spec.master_gb:g}GB" for spec in memory_model.models.values()
    )
    typer.echo(f"memory model masters (placeholders pending D-03): {masters}")
    budgeter = MemoryBudgeter(memory_model)
    residency = ResidencyManager(on_release=lambda lease: budgeter.release_delta(lease.lease_id))
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

    # Shared, read-only view of the run — used by the optional live dashboard
    # AND the optional showcase recorder, so both always agree with each
    # other and with /api/state's schema (dashboard.server.build_state_snapshot).
    sources = DashboardSources(
        event_log=orchestrator.event_log,
        metrics=metrics,
        memory_model=memory_model,
        residency=residency,
        engine_mode=engine_mode,
        hw_label=hw_label_active,
    )

    # --- optional L9 dashboard (read-only observation plane) ---
    dashboard_task: asyncio.Task | None = None
    if dashboard:
        dash_app = create_dashboard_app(sources)
        dash_config = uvicorn.Config(
            dash_app, host="127.0.0.1", port=dashboard_port, log_level="warning"
        )
        dash_server = uvicorn.Server(dash_config)
        dashboard_task = asyncio.create_task(dash_server.serve())
        typer.echo(f"Dashboard live at http://127.0.0.1:{dashboard_port} (read-only)")

    # --- optional showcase recorder — captures the boot snapshot now, before
    # any records have been replayed, exactly like a dashboard opened at t=0 ---
    recorder: RunRecorder | None = None
    if record:
        recorder = RunRecorder(sources=sources)
        recorder.snapshot()
        typer.echo("Recording this run for the static showcase build...")

    # --- replay + pipeline ---
    cases_processed = 0
    typer.echo(f"--- replaying at {speed}x ---")
    event_log = orchestrator.event_log
    last_snapshot_mono = time.monotonic()
    async for rec in stream(store.all_records(), speed_factor=speed):
        # Feed the observation plane: one "record" event per replayed record,
        # so every particle the L9 theater draws is a real Loghub line.
        event_log.append(
            "replay",
            "record",
            ts=rec.ts.isoformat(),
            node=rec.node,
            level=rec.level,
            component=rec.component,
            message=rec.message[:160],
            dialect=rec.dialect,
        )
        case = detector.observe(rec)
        if case is None:
            # Periodic snapshots (~poll cadence) so the showcase header/ledger
            # animate smoothly through the quiet stretches between cases.
            if recorder is not None and time.monotonic() - last_snapshot_mono >= 1.5:
                recorder.snapshot()
                last_snapshot_mono = time.monotonic()
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
        if recorder is not None:
            # Right after the verdict, so the recorded ledger's memory-cost
            # figures reflect the ledger at the moment this case closed.
            recorder.snapshot()
            last_snapshot_mono = time.monotonic()
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

    if export:
        cases = extract_cases_from_events(event_log.events(), snapshot["ledger"])
        report = build_incident_report(cases, snapshot, engine_mode=sources.engine_mode)
        csv_path, md_path = export_incident_report(report, Path("reports"))
        typer.echo("\nincident report exported:")
        typer.echo(f"  {csv_path}")
        typer.echo(f"  {md_path}")

    if recorder is not None:
        recorder.snapshot()
        meta = {
            "recorded_at": recorder.t0.isoformat(),
            "speed_factor": speed,
            "burst_n": burst_n,
            "burst_window_s": burst_window_s,
            "record_count": store.count(),
            "cases_processed": cases_processed,
            "engine_mode": sources.engine_mode,
            "hardware": sources.hardware_name,
            "source": str(log_file) if log_file is not None else str(db),
        }
        recording = recorder.build_recording(meta=meta)
        json_path = write_recording(recording)
        html_path = embed_recording_in_showcase(recording)
        typer.echo("\nshowcase recording written:")
        typer.echo(f"  {json_path}")
        typer.echo(f"  {html_path} (recording embedded — open it directly, no server needed)")

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
