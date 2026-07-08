"""
Layer L8 incident-report tests.

The report is the agents' actual deliverable: a file an on-call engineer
can hand to their team. Cases are run for real through the mock-engine
orchestrator (same fixture helpers as test_orchestrator.py) so the report
reflects the real schema-valid payloads, not a hand-rolled stand-in.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import httpx
import pytest

from app.orchestrator.report import (
    CSV_FIELDS,
    build_incident_report,
    export_incident_report,
    extract_cases_from_events,
    render_csv_report,
    render_markdown_report,
)
from core.budgeter import MemoryBudgeter
from core.memory_model import MemoryModel
from core.metrics import CoreMetrics
from deploy.mock_engine.main import app as mock_app
from tests.test_orchestrator import build_runtime, make_case

REQUIRED_ROW_KEYS = {
    "case_id",
    "opened_at",
    "severity",
    "title",
    "trigger_count",
    "nodes",
    "category",
    "root_cause",
    "evidence",
    "blast_radius_nodes",
    "blast_radius_edges",
    "blast_radius_labels",
    "recommendation",
    "contributing_agents",
    "degraded_agents",
    "working_delta_gb",
    "masters_gb_reused",
}


@pytest.fixture()
def transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=mock_app)


async def _run_case_and_metrics(transport, case_id: str):
    rt = build_runtime(transport)
    memory_model = MemoryModel()
    budgeter = MemoryBudgeter(memory_model)
    metrics = CoreMetrics(
        memory_model=memory_model,
        materializer=rt.materializer,
        residency=rt.residency,
        budgeter=budgeter,
    )
    await rt.orchestrator.run_case(make_case(case_id))
    return rt, metrics


class TestExtractCasesFromEvents:
    async def test_case_row_contains_every_column(self, transport) -> None:
        rt, metrics = await _run_case_and_metrics(transport, "case-report-1")
        ledger = metrics.export()["ledger"]

        cases = extract_cases_from_events(rt.orchestrator.event_log.events(), ledger)

        assert len(cases) == 1
        row = cases[0]
        assert REQUIRED_ROW_KEYS <= row.keys()
        assert row["case_id"] == "case-report-1"

        # Real values from the mock engine's deterministic role payloads.
        assert row["severity"] == "high"
        assert "network instability" in row["root_cause"]
        assert "Broken pipe" in row["evidence"]
        assert "NIC" in row["recommendation"]
        assert row["blast_radius_nodes"] == 2
        assert row["blast_radius_edges"] == 1
        assert row["degraded_agents"] == []
        assert row["contributing_agents"] == [
            "triage",
            "correlator",
            "hunter",
            "topology",
            "reporter",
        ]
        assert row["masters_gb_reused"] == ledger["masters_gb"]

    async def test_in_flight_case_without_verdict_is_excluded(self, transport) -> None:
        from app.orchestrator.orchestrator import EventLog

        log = EventLog()
        log.append("case-open-only", "case_open", opened_at="t0", trigger_records=[])
        log.append("case-open-only", "state", state="TRIAGE")

        cases = extract_cases_from_events(log.events())
        assert cases == []


class TestBuildIncidentReport:
    async def test_summary_fields(self, transport) -> None:
        rt, metrics = await _run_case_and_metrics(transport, "case-report-2")
        snapshot = metrics.export()
        cases = extract_cases_from_events(rt.orchestrator.event_log.events(), snapshot["ledger"])

        report = build_incident_report(cases, snapshot, engine_mode="mock")

        assert report["engine_mode"] == "mock"
        assert "generated_at" in report
        summary = report["summary"]
        assert summary["cases_processed"] == 1
        assert summary["prefills_avoided"] == snapshot["prefills_avoided"]
        assert summary["gb_saved_vs_per_agent"] == snapshot["gb_saved_vs_per_agent"]
        assert summary["evictions"] == 0
        assert summary["engine_mode"] == "mock"
        assert report["cases"] == cases


class TestRenderCsv:
    async def test_csv_parses_with_header_and_row_per_case(self, transport) -> None:
        rt, metrics = await _run_case_and_metrics(transport, "case-report-3")
        snapshot = metrics.export()
        cases = extract_cases_from_events(rt.orchestrator.event_log.events(), snapshot["ledger"])
        report = build_incident_report(cases, snapshot)

        text = render_csv_report(report)
        reader = csv.DictReader(io.StringIO(text))

        assert reader.fieldnames == CSV_FIELDS
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["case_id"] == "case-report-3"
        assert "network instability" in rows[0]["root_cause"]
        # list fields are flattened, never left as Python repr
        assert "[" not in rows[0]["nodes"]


class TestRenderMarkdown:
    async def test_markdown_contains_title_block_and_all_case_sections(self, transport) -> None:
        rt, metrics = await _run_case_and_metrics(transport, "case-report-4")
        # second case through the SAME orchestrator so both land in one report
        await rt.orchestrator.run_case(make_case("case-report-5"))
        snapshot = metrics.export()
        cases = extract_cases_from_events(rt.orchestrator.event_log.events(), snapshot["ledger"])
        assert len(cases) == 2
        report = build_incident_report(cases, snapshot, engine_mode="mock")

        md = render_markdown_report(report)

        assert md.startswith("# REMANON")
        assert "GPU-resident shared-memory runtime" in md
        assert "Engine mode:** mock" in md
        assert "## Run summary" in md
        for case in cases:
            assert f"## Case {case['case_id'][:8]}" in md
            assert case["root_cause"] in md
            assert case["recommendation"] in md


class TestExportIncidentReport:
    async def test_writes_both_files_with_matching_content(self, transport, tmp_path: Path) -> None:
        rt, metrics = await _run_case_and_metrics(transport, "case-report-6")
        snapshot = metrics.export()
        cases = extract_cases_from_events(rt.orchestrator.event_log.events(), snapshot["ledger"])
        report = build_incident_report(cases, snapshot, engine_mode="mock")

        csv_path, md_path = export_incident_report(report, tmp_path, timestamp="20260101T000000Z")

        assert csv_path.name == "incident_report_20260101T000000Z.csv"
        assert md_path.name == "incident_report_20260101T000000Z.md"
        # csv.writer's dialect always emits \r\n; read back with newline="" to
        # compare the exact bytes written, matching what Excel expects.
        with csv_path.open(encoding="utf-8", newline="") as f:
            assert f.read() == render_csv_report(report)
        assert md_path.read_text(encoding="utf-8") == render_markdown_report(report)
