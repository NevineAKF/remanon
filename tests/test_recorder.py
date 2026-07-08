"""
Layer L9 showcase-recorder tests.

Uses tmp_path for every write — never touches the real
dashboard/showcase/run_recording.json or index.html, which hold a real
generated recording checked in for the static showcase build.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from core.budgeter import MemoryBudgeter
from core.memory_model import MemoryModel
from core.metrics import CoreMetrics
from dashboard.recorder import RunRecorder, embed_recording_in_showcase, write_recording
from dashboard.server import DashboardSources
from deploy.mock_engine.main import app as mock_app
from tests.test_orchestrator import build_runtime, make_case

_DATA_ISLAND_STUB = (
    "<!doctype html><html><body>"
    '<script type="application/json" id="run-recording-data">{}</script>'
    "</body></html>"
)


@pytest.fixture()
def transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=mock_app)


def _sources(rt) -> DashboardSources:
    memory_model = MemoryModel()
    budgeter = MemoryBudgeter(memory_model)
    metrics = CoreMetrics(
        memory_model=memory_model,
        materializer=rt.materializer,
        residency=rt.residency,
        budgeter=budgeter,
    )
    return DashboardSources(
        event_log=rt.orchestrator.event_log,
        metrics=metrics,
        memory_model=memory_model,
        residency=rt.residency,
    )


class TestRunRecorder:
    async def test_snapshots_and_events_have_nonnegative_increasing_t_rel(self, transport) -> None:
        rt = build_runtime(transport)
        sources = _sources(rt)
        recorder = RunRecorder(sources=sources)

        recorder.snapshot()  # boot
        await rt.orchestrator.run_case(make_case("case-rec-1"))
        recorder.snapshot()  # post-verdict

        recording = recorder.build_recording(meta={"speed_factor": 1000.0})

        assert len(recording["snapshots"]) == 2
        t_rels = [s["t_rel"] for s in recording["snapshots"]]
        assert t_rels[0] >= 0
        assert t_rels == sorted(t_rels)

        assert recording["meta"] == {"speed_factor": 1000.0}
        assert len(recording["events"]) == len(sources.event_log.events())
        for event in recording["events"]:
            assert event["t_rel"] >= 0
            assert "kind" in event and "case_id" in event and "data" in event

    async def test_snapshot_state_matches_live_schema(self, transport) -> None:
        rt = build_runtime(transport)
        sources = _sources(rt)
        recorder = RunRecorder(sources=sources)
        recorder.snapshot()

        recording = recorder.build_recording(meta={})
        state = recording["snapshots"][0]["state"]
        for key in (
            "metrics",
            "capacity_gb",
            "headroom_gb",
            "hardware",
            "memory_tech",
            "engine_mode",
            "masters_config",
            "active_leases",
            "cases_processed",
            "events",
            "verdicts",
        ):
            assert key in state, f"missing key: {key}"


class TestWriteRecording:
    async def test_writes_valid_json(self, transport, tmp_path: Path) -> None:
        rt = build_runtime(transport)
        sources = _sources(rt)
        recorder = RunRecorder(sources=sources)
        recorder.snapshot()
        recording = recorder.build_recording(meta={"engine_mode": "mock"})

        out_path = write_recording(recording, tmp_path / "run_recording.json")

        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded == recording


class TestEmbedRecordingInShowcase:
    def test_splices_recording_into_data_island(self, tmp_path: Path) -> None:
        html_path = tmp_path / "index.html"
        html_path.write_text(_DATA_ISLAND_STUB, encoding="utf-8")
        recording = {"meta": {"engine_mode": "mock"}, "events": [], "snapshots": []}

        result_path = embed_recording_in_showcase(recording, html_path)

        assert result_path == html_path
        html = html_path.read_text(encoding="utf-8")
        assert '<script type="application/json" id="run-recording-data">' in html
        assert json.dumps(recording, separators=(",", ":")) in html
        # surrounding markup untouched
        assert html.startswith("<!doctype html>")
        assert html.endswith("</html>")

    def test_escapes_closing_script_tag_in_recorded_strings(self, tmp_path: Path) -> None:
        html_path = tmp_path / "index.html"
        html_path.write_text(_DATA_ISLAND_STUB, encoding="utf-8")
        recording = {"meta": {}, "events": [{"data": {"message": "</script><script>evil()"}}]}

        embed_recording_in_showcase(recording, html_path)

        html = html_path.read_text(encoding="utf-8")
        assert "</script><script>evil()" not in html
        # the payload must still round-trip to the exact original recording
        import re

        m = re.search(
            r'<script type="application/json" id="run-recording-data">(.*?)</script>', html, re.S
        )
        assert json.loads(m.group(1)) == recording

    def test_missing_html_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            embed_recording_in_showcase({"meta": {}}, tmp_path / "missing.html")

    def test_missing_data_island_raises(self, tmp_path: Path) -> None:
        html_path = tmp_path / "index.html"
        html_path.write_text(
            "<!doctype html><html><body>no island here</body></html>", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="data island"):
            embed_recording_in_showcase({"meta": {}}, html_path)
