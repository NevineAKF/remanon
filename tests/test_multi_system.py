"""
Multi-system ingest tests — Layer L5.

Covers the new real Loghub dialects (BGL, Thunderbird, Spark, Hadoop) and
the combined multi-system store/replay path. Uses small real-line fixture
excerpts (tests/fixtures/*_sample.log, extracted verbatim from the real
Loghub 2k samples) — no network calls, no synthetic data.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from app.dataplane import ingest as ingest_module
from app.dataplane.normalizer import get_normalizer
from app.dataplane.parser import parse_file, read_raw_lines
from app.dataplane.replayer import stream
from app.dataplane.store import TelemetryStore

_FIXTURES = Path("tests/fixtures")

_FIXTURE_MAP = {
    "hdfs_2k": "hdfs_sample.log",
    "bgl_2k": "bgl_sample.log",
    "thunderbird_2k": "thunderbird_sample.log",
    "spark_2k": "spark_sample.log",
    "hadoop_2k": "hadoop_sample.log",
}


def _combined_records() -> list:
    records = []
    for dialect, fname in [
        ("hdfs", "hdfs_sample.log"),
        ("bgl", "bgl_sample.log"),
        ("thunderbird", "thunderbird_sample.log"),
        ("spark", "spark_sample.log"),
        ("hadoop", "hadoop_sample.log"),
    ]:
        norm = get_normalizer(dialect)
        if dialect == "hdfs":
            parsed = parse_file(_FIXTURES / fname)
            records += [norm.normalize(p) for p in parsed.records]
        else:
            lines = read_raw_lines(_FIXTURES / fname)
            records += [r for line in lines if (r := norm.parse_raw(line)) is not None]
    return records


# ---------------------------------------------------------------------------
# BGL
# ---------------------------------------------------------------------------


class TestBGLNormalizer:
    def test_parses_every_real_line(self) -> None:
        lines = read_raw_lines(_FIXTURES / "bgl_sample.log")
        norm = get_normalizer("bgl")
        records = [norm.parse_raw(line) for line in lines]
        assert all(r is not None for r in records)
        assert all(r.dialect == "bgl" for r in records)

    def test_real_level_annotations_map_correctly(self) -> None:
        lines = read_raw_lines(_FIXTURES / "bgl_sample.log")
        norm = get_normalizer("bgl")
        records = [norm.parse_raw(line) for line in lines]
        levels = {r.level for r in records}
        # fixture carries real INFO/FATAL/WARNING/ERROR annotations
        assert {"INFO", "FATAL", "WARN", "ERROR"} <= levels

    def test_ts_from_real_embedded_epoch(self) -> None:
        lines = read_raw_lines(_FIXTURES / "bgl_sample.log")
        norm = get_normalizer("bgl")
        rec = norm.parse_raw(lines[0])
        assert rec is not None
        assert rec.ts.year == 2005

    def test_unmatched_line_returns_none_not_fabricated(self) -> None:
        norm = get_normalizer("bgl")
        assert norm.parse_raw("not a bgl-shaped line at all") is None


# ---------------------------------------------------------------------------
# Thunderbird
# ---------------------------------------------------------------------------


class TestThunderbirdNormalizer:
    def test_parses_every_real_line(self) -> None:
        lines = read_raw_lines(_FIXTURES / "thunderbird_sample.log")
        norm = get_normalizer("thunderbird")
        records = [norm.parse_raw(line) for line in lines]
        assert all(r is not None for r in records)
        assert all(r.dialect == "thunderbird" for r in records)

    def test_content_derived_severity_from_real_text(self) -> None:
        """The 2k sample carries no Loghub anomaly-label lines, so severity
        must come from real message text ('Wait for ready failed...')."""
        lines = read_raw_lines(_FIXTURES / "thunderbird_sample.log")
        norm = get_normalizer("thunderbird")
        records = [norm.parse_raw(line) for line in lines]
        assert any(r.level == "ERROR" for r in records)
        assert any(r.level == "INFO" for r in records)

    def test_ts_from_real_embedded_epoch(self) -> None:
        lines = read_raw_lines(_FIXTURES / "thunderbird_sample.log")
        norm = get_normalizer("thunderbird")
        rec = norm.parse_raw(lines[0])
        assert rec is not None
        assert rec.ts.year == 2005


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------


class TestSparkNormalizer:
    def test_parses_every_real_line(self) -> None:
        lines = read_raw_lines(_FIXTURES / "spark_sample.log")
        norm = get_normalizer("spark")
        records = [norm.parse_raw(line) for line in lines]
        assert all(r is not None for r in records)
        assert all(r.dialect == "spark" for r in records)

    def test_real_sample_is_honestly_all_info(self) -> None:
        # The real Spark_2k.log sample genuinely contains zero WARN/ERROR
        # lines — asserting that honestly rather than inventing alerts.
        lines = read_raw_lines(_FIXTURES / "spark_sample.log")
        norm = get_normalizer("spark")
        records = [norm.parse_raw(line) for line in lines]
        assert all(r.level == "INFO" for r in records)

    def test_ts_from_real_date(self) -> None:
        lines = read_raw_lines(_FIXTURES / "spark_sample.log")
        norm = get_normalizer("spark")
        rec = norm.parse_raw(lines[0])
        assert rec is not None
        assert rec.ts.year == 2017


# ---------------------------------------------------------------------------
# Hadoop
# ---------------------------------------------------------------------------


class TestHadoopNormalizer:
    def test_parses_every_real_line(self) -> None:
        lines = read_raw_lines(_FIXTURES / "hadoop_sample.log")
        norm = get_normalizer("hadoop")
        records = [norm.parse_raw(line) for line in lines]
        assert all(r is not None for r in records)
        assert all(r.dialect == "hadoop" for r in records)

    def test_real_warn_error_fatal_present(self) -> None:
        lines = read_raw_lines(_FIXTURES / "hadoop_sample.log")
        norm = get_normalizer("hadoop")
        records = [norm.parse_raw(line) for line in lines]
        levels = {r.level for r in records}
        assert {"WARN", "ERROR", "FATAL"} <= levels

    def test_node_extracted_from_real_ip_in_message(self) -> None:
        lines = read_raw_lines(_FIXTURES / "hadoop_sample.log")
        norm = get_normalizer("hadoop")
        records = [norm.parse_raw(line) for line in lines]
        assert any(r.node != "unknown" for r in records)


# ---------------------------------------------------------------------------
# Combined multi-system store + replay
# ---------------------------------------------------------------------------


class TestCombinedMultiSystemStore:
    def test_dialect_column_distinguishes_system(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "combined.duckdb")
        store.write_records(_combined_records())
        systems = {row["dialect"] for row in store.query("SELECT DISTINCT dialect FROM telemetry")}
        assert systems == {"hdfs", "bgl", "thunderbird", "spark", "hadoop"}
        store.close()

    def test_all_records_true_chronological_order_across_systems(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "combined.duckdb")
        store.write_records(_combined_records())
        ordered = store.all_records()
        assert {r.dialect for r in ordered} == {"hdfs", "bgl", "thunderbird", "spark", "hadoop"}
        for a, b in zip(ordered, ordered[1:], strict=False):
            assert a.ts <= b.ts
        store.close()

    def test_replayer_handles_huge_cross_system_gaps_with_cap(self) -> None:
        """BGL/Thunderbird (2005) to Spark (2017) spans ~12 real years —
        the max_sleep_s cap must keep replay of the combined store fast,
        exactly like it already does for a single system's internal gaps."""
        records = _combined_records()
        collected: list = []

        async def _run() -> None:
            async for rec in stream(records, speed_factor=1.0, max_sleep_s=0.01):
                collected.append(rec)

        asyncio.run(_run())
        assert len(collected) == len(records)
        for a, b in zip(collected, collected[1:], strict=False):
            assert a.ts <= b.ts


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestIngestRegistry:
    def test_registry_has_all_five_datasets_with_correct_dialects(self) -> None:
        expected = {
            "hdfs_2k": "hdfs",
            "bgl_2k": "bgl",
            "thunderbird_2k": "thunderbird",
            "spark_2k": "spark",
            "hadoop_2k": "hadoop",
        }
        assert set(ingest_module._DATASETS) == set(expected)
        for key, dialect in expected.items():
            meta = ingest_module._DATASETS[key]
            assert meta["dialect"] == dialect
            assert meta["url"] == (
                f"https://raw.githubusercontent.com/logpai/loghub/master/"
                f"{meta['filename'].split('_')[0]}/{meta['filename']}"
            )


# ---------------------------------------------------------------------------
# CLI — driven against cached (pre-placed) real fixture files, no network
# ---------------------------------------------------------------------------


class TestIngestCLI:
    @pytest.fixture()
    def isolated_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        raw_dir = tmp_path / "raw"
        store_path = tmp_path / "store" / "telemetry.duckdb"
        monkeypatch.setattr(ingest_module, "_RAW_DIR", raw_dir)
        monkeypatch.setattr(ingest_module, "_STORE_PATH", store_path)
        raw_dir.mkdir(parents=True)
        for key, fname in _FIXTURE_MAP.items():
            meta = ingest_module._DATASETS[key]
            (raw_dir / meta["filename"]).write_bytes((_FIXTURES / fname).read_bytes())
        return store_path

    def test_dataset_all_writes_every_system_and_reports_counts(self, isolated_store: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(ingest_module.app, ["--dataset", "all"])
        assert result.exit_code == 0, result.output
        for key in _FIXTURE_MAP:
            assert key in result.output
        assert "Combined store total" in result.output
        assert "Combined time span" in result.output

        store = TelemetryStore(isolated_store)
        systems = {row["dialect"] for row in store.query("SELECT DISTINCT dialect FROM telemetry")}
        assert systems == {"hdfs", "bgl", "thunderbird", "spark", "hadoop"}
        store.close()

    def test_single_dataset_still_works(self, isolated_store: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(ingest_module.app, ["--dataset", "hdfs_2k"])
        assert result.exit_code == 0, result.output
        store = TelemetryStore(isolated_store)
        systems = {row["dialect"] for row in store.query("SELECT DISTINCT dialect FROM telemetry")}
        assert systems == {"hdfs"}
        store.close()

    def test_download_failure_is_skipped_not_fabricated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_dir = tmp_path / "raw"
        store_path = tmp_path / "store" / "telemetry.duckdb"
        monkeypatch.setattr(ingest_module, "_RAW_DIR", raw_dir)
        monkeypatch.setattr(ingest_module, "_STORE_PATH", store_path)

        def _boom(url: str, dest: Path) -> None:
            raise httpx.ConnectError(
                "simulated download failure", request=httpx.Request("GET", url)
            )

        monkeypatch.setattr(ingest_module, "_download", _boom)
        runner = CliRunner()
        result = runner.invoke(ingest_module.app, ["--dataset", "hdfs_2k"])
        assert result.exit_code == 1
        assert "SKIPPED" in result.output
        assert not store_path.exists()

    def test_dataset_all_skips_failed_download_but_keeps_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_dir = tmp_path / "raw"
        store_path = tmp_path / "store" / "telemetry.duckdb"
        monkeypatch.setattr(ingest_module, "_RAW_DIR", raw_dir)
        monkeypatch.setattr(ingest_module, "_STORE_PATH", store_path)
        raw_dir.mkdir(parents=True)
        # Pre-cache every dataset EXCEPT hdfs_2k, whose download will fail.
        for key, fname in _FIXTURE_MAP.items():
            if key == "hdfs_2k":
                continue
            meta = ingest_module._DATASETS[key]
            (raw_dir / meta["filename"]).write_bytes((_FIXTURES / fname).read_bytes())

        real_download = ingest_module._download

        def _selective_boom(url: str, dest: Path) -> None:
            if "HDFS" in url:
                raise httpx.ConnectError("simulated 404", request=httpx.Request("GET", url))
            real_download(url, dest)

        monkeypatch.setattr(ingest_module, "_download", _selective_boom)
        runner = CliRunner()
        result = runner.invoke(ingest_module.app, ["--dataset", "all"])
        assert result.exit_code == 0, result.output
        assert "hdfs_2k" in result.output and "SKIPPED" in result.output

        store = TelemetryStore(store_path)
        systems = {row["dialect"] for row in store.query("SELECT DISTINCT dialect FROM telemetry")}
        assert systems == {"bgl", "thunderbird", "spark", "hadoop"}  # never fabricated hdfs rows
        store.close()
