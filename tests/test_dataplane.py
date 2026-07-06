"""
Data-plane tests — Layer L5.

No network calls.  All tests use the 50-line HDFS fixture at
tests/fixtures/hdfs_sample.log.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.dataplane.normalizer import HDFSNormalizer, TelemetryRecord, _parse_hdfs_ts
from app.dataplane.parser import ParsedLine, ParseResult, parse_file, parse_lines
from app.dataplane.replayer import compute_wall_duration, stream
from app.dataplane.store import TelemetryStore

_FIXTURE = Path("tests/fixtures/hdfs_sample.log")


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parsed() -> ParseResult:
    return parse_file(_FIXTURE)


@pytest.fixture(scope="module")
def records(parsed: ParseResult) -> list[TelemetryRecord]:
    norm = HDFSNormalizer()
    return [norm.normalize(p) for p in parsed.records]


@pytest.fixture()
def tmp_store(tmp_path: Path) -> TelemetryStore:
    return TelemetryStore(tmp_path / "test.duckdb")


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_skips_malformed_line(self, parsed: ParseResult) -> None:
        assert parsed.skipped >= 1, "Expected at least one malformed line to be skipped"

    def test_parses_majority_of_lines(self, parsed: ParseResult) -> None:
        assert len(parsed.records) >= 48

    def test_level_values_are_uppercase(self, parsed: ParseResult) -> None:
        for rec in parsed.records:
            assert rec.level == rec.level.upper()

    def test_parsed_line_fields(self, parsed: ParseResult) -> None:
        first = parsed.records[0]
        assert isinstance(first, ParsedLine)
        assert first.date_str == "081109"
        assert first.time_str == "203518"
        assert first.level == "INFO"
        assert "DataXceiver" in first.component or "FSNamesystem" in first.component

    def test_warn_and_error_present(self, parsed: ParseResult) -> None:
        levels = {r.level for r in parsed.records}
        assert "WARN" in levels
        assert "ERROR" in levels

    def test_empty_input_yields_no_records(self) -> None:
        result = parse_lines(iter([]))
        assert result.records == []
        assert result.skipped == 0

    def test_single_malformed_line(self) -> None:
        result = parse_lines(iter(["this is not a valid hdfs log line"]))
        assert result.records == []
        assert result.skipped == 1


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------


class TestNormalizer:
    def test_ts_is_utc_aware(self, records: list[TelemetryRecord]) -> None:
        for r in records:
            assert r.ts.tzinfo is not None

    def test_ts_year_is_2008(self, records: list[TelemetryRecord]) -> None:
        assert all(r.ts.year == 2008 for r in records)

    def test_node_is_non_empty(self, records: list[TelemetryRecord]) -> None:
        for r in records:
            assert r.node, "node must be a non-empty string"

    def test_dialect_is_hdfs(self, records: list[TelemetryRecord]) -> None:
        assert all(r.dialect == "hdfs" for r in records)

    def test_parse_hdfs_ts(self) -> None:
        ts = _parse_hdfs_ts("081109", "203518")
        assert ts == datetime(2008, 11, 9, 20, 35, 18, tzinfo=UTC)

    def test_record_is_frozen(self, records: list[TelemetryRecord]) -> None:
        r = records[0]
        with pytest.raises(AttributeError):
            r.level = "MUTATED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------


class TestTelemetryStore:
    def test_empty_store_count_is_zero(self, tmp_store: TelemetryStore) -> None:
        assert tmp_store.count() == 0

    def test_write_and_count(
        self, tmp_store: TelemetryStore, records: list[TelemetryRecord]
    ) -> None:
        tmp_store.write_records(records)
        assert tmp_store.count() == len(records)

    def test_time_range_none_on_empty_store(self, tmp_store: TelemetryStore) -> None:
        assert tmp_store.time_range() is None

    def test_time_range_after_write(
        self, tmp_store: TelemetryStore, records: list[TelemetryRecord]
    ) -> None:
        tmp_store.write_records(records)
        span = tmp_store.time_range()
        assert span is not None
        lo, hi = span
        assert lo <= hi

    def test_sql_query_filter_by_level(
        self, tmp_store: TelemetryStore, records: list[TelemetryRecord]
    ) -> None:
        tmp_store.write_records(records)
        errors = tmp_store.query("SELECT * FROM telemetry WHERE level = 'ERROR'")
        assert len(errors) >= 1
        assert all(row["level"] == "ERROR" for row in errors)

    def test_sql_query_count_warn(
        self, tmp_store: TelemetryStore, records: list[TelemetryRecord]
    ) -> None:
        tmp_store.write_records(records)
        rows = tmp_store.query("SELECT COUNT(*) AS n FROM telemetry WHERE level = 'WARN'")
        assert rows[0]["n"] >= 1

    def test_write_empty_list_is_noop(self, tmp_store: TelemetryStore) -> None:
        tmp_store.write_records([])
        assert tmp_store.count() == 0

    def test_all_records_sorted_ascending(
        self, tmp_store: TelemetryStore, records: list[TelemetryRecord]
    ) -> None:
        # Shuffle insertion order
        shuffled = sorted(records, key=lambda r: r.node)
        tmp_store.write_records(shuffled)
        result = tmp_store.all_records()
        for a, b in zip(result, result[1:], strict=False):
            assert a.ts <= b.ts

    def test_to_parquet(
        self, tmp_store: TelemetryStore, records: list[TelemetryRecord], tmp_path: Path
    ) -> None:
        tmp_store.write_records(records)
        parquet_path = tmp_store.to_parquet(tmp_path / "snap.parquet")
        assert parquet_path.exists()
        assert parquet_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Replayer tests
# ---------------------------------------------------------------------------


class TestReplayer:
    def test_records_emitted_in_chronological_order(self, records: list[TelemetryRecord]) -> None:
        # Shuffle before passing to replayer
        shuffled = sorted(records, key=lambda r: r.node)
        collected: list[TelemetryRecord] = []

        async def _run() -> None:
            # Use enormous speed_factor so sleeps are zero-length
            async for rec in stream(shuffled, speed_factor=1_000_000_000.0):
                collected.append(rec)

        asyncio.run(_run())
        for a, b in zip(collected, collected[1:], strict=False):
            assert a.ts <= b.ts

    def test_all_records_emitted(self, records: list[TelemetryRecord]) -> None:
        collected: list[TelemetryRecord] = []

        async def _run() -> None:
            async for rec in stream(records, speed_factor=1_000_000_000.0):
                collected.append(rec)

        asyncio.run(_run())
        assert len(collected) == len(records)

    def test_empty_records_emits_nothing(self) -> None:
        collected: list[TelemetryRecord] = []

        async def _run() -> None:
            async for rec in stream([], speed_factor=60.0):
                collected.append(rec)

        asyncio.run(_run())
        assert collected == []

    def test_callback_is_invoked_for_each_record(self, records: list[TelemetryRecord]) -> None:
        invoked: list[TelemetryRecord] = []

        async def cb(rec: TelemetryRecord) -> None:
            invoked.append(rec)

        async def _run() -> None:
            async for _ in stream(records, speed_factor=1_000_000_000.0, callback=cb):
                pass

        asyncio.run(_run())
        assert len(invoked) == len(records)

    def test_compute_wall_duration_compression_math(self, records: list[TelemetryRecord]) -> None:
        # With speed_factor=60, wall duration should be log_span / 60
        sorted_recs = sorted(records, key=lambda r: r.ts)
        log_span_s = (sorted_recs[-1].ts - sorted_recs[0].ts).total_seconds()
        # compute_wall_duration sums per-gap sleeps (capped at max_sleep_s=5)
        wall = compute_wall_duration(records, speed_factor=60.0, max_sleep_s=3600.0)
        # Without the cap the wall time should equal log_span / speed_factor
        assert abs(wall - log_span_s / 60.0) < 0.001

    def test_max_sleep_caps_large_gap(self) -> None:
        # Two records 1 hour apart, speed_factor=1 → 3600s gap, capped at 0.1s
        r1 = TelemetryRecord(
            ts=datetime(2008, 11, 9, 20, 0, 0, tzinfo=UTC),
            node="10.0.0.1",
            level="INFO",
            component="A",
            message="m1",
            dialect="hdfs",
            raw_line="",
        )
        r2 = TelemetryRecord(
            ts=datetime(2008, 11, 9, 21, 0, 0, tzinfo=UTC),
            node="10.0.0.1",
            level="INFO",
            component="A",
            message="m2",
            dialect="hdfs",
            raw_line="",
        )
        wall = compute_wall_duration([r1, r2], speed_factor=1.0, max_sleep_s=0.1)
        assert wall == pytest.approx(0.1)
