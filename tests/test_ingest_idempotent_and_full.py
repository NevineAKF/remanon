"""
Ingest idempotency + full-size Loghub tests — Layer L5.

Part 1: TelemetryStore.write_records is idempotent per dialect (delete
existing rows for that dialect, then insert) — re-running ingest any
number of times never accumulates duplicates. Part 2: the registry's
full_url entries (hadoop/bgl/hdfs, verified live against Zenodo record
8196385) and the --size full / --max-records extraction path.

No real network: --download is monkeypatched to copy a small, LOCALLY
built archive (containing real fixture lines) instead of hitting Zenodo,
and 2k-mode tests use the same cached-file pattern as test_multi_system.py.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.dataplane import ingest as ingest_module
from app.dataplane.normalizer import TelemetryRecord
from app.dataplane.store import TelemetryStore
from tests.test_multi_system import _FIXTURE_MAP, _FIXTURES


def _rec(dialect: str, node: str = "n1") -> TelemetryRecord:
    from datetime import UTC, datetime

    return TelemetryRecord(
        ts=datetime(2020, 1, 1, tzinfo=UTC),
        node=node,
        level="INFO",
        component="c",
        message="m",
        dialect=dialect,
        raw_line="raw",
    )


# ---------------------------------------------------------------------------
# Part 1 — TelemetryStore.write_records idempotency + reset()
# ---------------------------------------------------------------------------


class TestWriteRecordsIdempotent:
    def test_rewriting_same_dialect_replaces_not_accumulates(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "t.duckdb")
        batch = [_rec("hdfs", f"n{i}") for i in range(5)]

        store.write_records(batch)
        assert store.count() == 5

        store.write_records(batch)  # re-ingest the SAME dataset again
        assert store.count() == 5  # not 10

        store.write_records(batch)  # and a third time
        assert store.count() == 5  # still 5, never grows
        store.close()

    def test_rewriting_different_dialects_do_not_clobber_each_other(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "t.duckdb")
        store.write_records([_rec("hdfs", "a"), _rec("hdfs", "b")])
        store.write_records([_rec("bgl", "c")])
        assert store.count() == 3

        # Re-ingesting hdfs must not touch bgl's rows.
        store.write_records([_rec("hdfs", "a"), _rec("hdfs", "b"), _rec("hdfs", "new")])
        assert store.count() == 4  # 3 hdfs (replaced) + 1 bgl (untouched)
        rows = store.query("SELECT dialect, COUNT(*) AS n FROM telemetry GROUP BY dialect")
        counts = {r["dialect"]: r["n"] for r in rows}
        assert counts == {"hdfs": 3, "bgl": 1}
        store.close()

    def test_mixed_dialect_batch_replaces_each_dialect_independently(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "t.duckdb")
        mixed = [_rec("hdfs", "a"), _rec("bgl", "b"), _rec("spark", "c")]
        store.write_records(mixed)
        store.write_records(mixed)  # re-run the exact same combined write
        assert store.count() == 3
        store.close()

    def test_write_empty_list_is_still_a_noop(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "t.duckdb")
        store.write_records([_rec("hdfs")])
        store.write_records([])  # must not touch existing hdfs rows
        assert store.count() == 1
        store.close()

    def test_reset_truncates_everything(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "t.duckdb")
        store.write_records([_rec("hdfs"), _rec("bgl"), _rec("spark")])
        assert store.count() == 3
        store.reset()
        assert store.count() == 0
        store.close()


# ---------------------------------------------------------------------------
# Part 1 — ingest CLI: --reset, and re-running never grows the store
# ---------------------------------------------------------------------------


class TestIngestCliIdempotent:
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

    def test_rerunning_dataset_all_does_not_grow_row_count(self, isolated_store: Path) -> None:
        runner = CliRunner()
        first = runner.invoke(ingest_module.app, ["--dataset", "all"])
        assert first.exit_code == 0, first.output
        store = TelemetryStore(isolated_store)
        count_after_first = store.count()
        store.close()
        assert count_after_first > 0

        second = runner.invoke(ingest_module.app, ["--dataset", "all"])
        assert second.exit_code == 0, second.output
        store = TelemetryStore(isolated_store)
        count_after_second = store.count()
        store.close()

        assert count_after_second == count_after_first  # never grows

        third = runner.invoke(ingest_module.app, ["--dataset", "all"])
        assert third.exit_code == 0, third.output
        store = TelemetryStore(isolated_store)
        count_after_third = store.count()
        store.close()
        assert count_after_third == count_after_first  # still stable

    def test_reset_flag_truncates_leftover_dialects_first(
        self, isolated_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate leftover data from a dataset not part of this run.
        store = TelemetryStore(isolated_store)
        store.write_records([_rec("some-old-dialect")])
        store.close()

        runner = CliRunner()
        result = runner.invoke(ingest_module.app, ["--dataset", "hdfs_2k", "--reset"])
        assert result.exit_code == 0, result.output

        store = TelemetryStore(isolated_store)
        dialects = {r["dialect"] for r in store.query("SELECT DISTINCT dialect FROM telemetry")}
        store.close()
        assert dialects == {"hdfs"}  # the leftover dialect is gone


# ---------------------------------------------------------------------------
# Part 2 — registry: full_url for hadoop/bgl/hdfs, absent for spark/thunderbird
# ---------------------------------------------------------------------------


class TestFullUrlRegistry:
    def test_full_url_present_for_hadoop_bgl_hdfs(self) -> None:
        zenodo_base = "https://zenodo.org/records/8196385/files"
        assert (
            ingest_module._DATASETS["hadoop_2k"]["full_url"]
            == f"{zenodo_base}/Hadoop.zip?download=1"
        )
        assert ingest_module._DATASETS["bgl_2k"]["full_url"] == f"{zenodo_base}/BGL.zip?download=1"
        assert (
            ingest_module._DATASETS["hdfs_2k"]["full_url"]
            == f"{zenodo_base}/HDFS_v1.zip?download=1"
        )

    def test_full_url_absent_for_spark_and_thunderbird(self) -> None:
        assert "full_url" not in ingest_module._DATASETS["spark_2k"]
        assert "full_url" not in ingest_module._DATASETS["thunderbird_2k"]

    def test_registry_still_has_all_five_2k_urls_unchanged(self) -> None:
        for key in ("hdfs_2k", "bgl_2k", "thunderbird_2k", "spark_2k", "hadoop_2k"):
            meta = ingest_module._DATASETS[key]
            assert meta["url"].startswith("https://raw.githubusercontent.com/logpai/loghub/master/")
            assert meta["url"].endswith(meta["filename"])


# ---------------------------------------------------------------------------
# Part 2 — --size full falls back to 2k (with a note) when no full_url
# ---------------------------------------------------------------------------


class TestSizeFullFallback:
    @pytest.fixture()
    def isolated_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        raw_dir = tmp_path / "raw"
        store_path = tmp_path / "store" / "telemetry.duckdb"
        monkeypatch.setattr(ingest_module, "_RAW_DIR", raw_dir)
        monkeypatch.setattr(ingest_module, "_STORE_PATH", store_path)
        raw_dir.mkdir(parents=True)
        meta = ingest_module._DATASETS["spark_2k"]
        (raw_dir / meta["filename"]).write_bytes((_FIXTURES / "spark_sample.log").read_bytes())
        return store_path

    def test_size_full_falls_back_to_2k_for_spark_with_a_note(self, isolated_store: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(ingest_module.app, ["--dataset", "spark_2k", "--size", "full"])
        assert result.exit_code == 0, result.output
        assert "NOTE" in result.output
        assert "falling back to the 2k sample" in result.output

        store = TelemetryStore(isolated_store)
        assert store.count() > 0  # the 2k fixture WAS ingested, not skipped
        store.close()

    def test_unknown_size_flag_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(ingest_module.app, ["--dataset", "hdfs_2k", "--size", "huge"])
        assert result.exit_code == 1
        assert "Unknown --size" in result.output


# ---------------------------------------------------------------------------
# Part 2 — full-archive extraction: real content, multi-file concatenation,
# non-.log entries excluded, archive cleaned up. No network: a small local
# zip built from real fixture lines stands in for the Zenodo download.
# ---------------------------------------------------------------------------


class TestExtractFullArchive:
    @pytest.fixture()
    def isolated_raw_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        raw_dir = tmp_path / "raw"
        monkeypatch.setattr(ingest_module, "_RAW_DIR", raw_dir)
        return raw_dir

    @pytest.fixture()
    def fake_hadoop_archive(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Builds a small zip shaped like the REAL Hadoop.zip (multiple real
        per-container .log files in subdirectories + a non-.log entry), and
        makes ingest_module._download 'download' it instead of hitting Zenodo."""
        real_lines = (_FIXTURES / "hadoop_sample.log").read_text(encoding="utf-8").splitlines()
        archive_src = tmp_path / "source_archive.zip"
        with zipfile.ZipFile(archive_src, "w") as zf:
            zf.writestr("application_1/container_2.log", "\n".join(real_lines[3:]) + "\n")
            zf.writestr("application_1/container_1.log", "\n".join(real_lines[:3]) + "\n")
            zf.writestr("README.md", "not a log file — must be excluded")

        def fake_download(url: str, dest: Path) -> None:
            dest.write_bytes(archive_src.read_bytes())

        monkeypatch.setattr(ingest_module, "_download", fake_download)

    async def test_extracts_and_concatenates_real_logs_in_sorted_path_order(
        self, isolated_raw_dir: Path, fake_hadoop_archive: None
    ) -> None:
        meta = dict(ingest_module._DATASETS["hadoop_2k"])
        result_path = ingest_module._download_and_extract_full("hadoop_2k", meta, force=False)

        assert result_path is not None
        assert result_path == isolated_raw_dir / "hadoop_full.log"
        content = result_path.read_text(encoding="utf-8")

        real_lines = (_FIXTURES / "hadoop_sample.log").read_text(encoding="utf-8").splitlines()
        # container_1.log sorts before container_2.log -> its lines come first,
        # even though it was written second into the zip.
        expected = "\n".join(real_lines[:3]) + "\n" + "\n".join(real_lines[3:]) + "\n"
        assert content == expected
        assert "not a log file" not in content  # README.md correctly excluded

        # The archive itself must not be left lying around anywhere under raw/.
        assert not list(isolated_raw_dir.glob("*.zip"))

    async def test_cached_full_file_is_reused_without_redownloading(
        self, isolated_raw_dir: Path, fake_hadoop_archive: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        meta = dict(ingest_module._DATASETS["hadoop_2k"])
        first = ingest_module._download_and_extract_full("hadoop_2k", meta, force=False)
        assert first is not None

        def fail_if_called(url: str, dest: Path) -> None:
            raise AssertionError("should not re-download when the extracted file is cached")

        monkeypatch.setattr(ingest_module, "_download", fail_if_called)
        second = ingest_module._download_and_extract_full("hadoop_2k", meta, force=False)
        assert second == first


# ---------------------------------------------------------------------------
# Part 2 — --max-records caps a REAL prefix, never fabricates
# ---------------------------------------------------------------------------


class TestMaxRecordsCap:
    def test_caps_to_a_real_prefix_of_hadoop_lines(self) -> None:
        records = ingest_module._parse_dataset(
            "hadoop_2k", "hadoop", _FIXTURES / "hadoop_sample.log", max_records=2
        )
        assert len(records) == 2
        # Must be the first two REAL parseable lines, not arbitrary/invented ones.
        full = ingest_module._parse_dataset(
            "hadoop_2k", "hadoop", _FIXTURES / "hadoop_sample.log", max_records=None
        )
        assert records == full[:2]

    def test_no_cap_returns_every_real_parseable_line(self) -> None:
        capped_huge = ingest_module._parse_dataset(
            "hadoop_2k", "hadoop", _FIXTURES / "hadoop_sample.log", max_records=10_000
        )
        uncapped = ingest_module._parse_dataset(
            "hadoop_2k", "hadoop", _FIXTURES / "hadoop_sample.log", max_records=None
        )
        assert capped_huge == uncapped

    def test_caps_hdfs_path_too(self) -> None:
        records = ingest_module._parse_dataset(
            "hdfs_2k", "hdfs", Path("tests/fixtures/hdfs_sample.log"), max_records=3
        )
        assert len(records) == 3
