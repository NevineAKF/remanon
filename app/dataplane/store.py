"""
TelemetryStore — Band A, Layer L5, decision D-01.

Persistent DuckDB database under data/store/telemetry.duckdb.
Parquet snapshots can be exported on demand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from app.dataplane.normalizer import TelemetryRecord

_DEFAULT_DB = Path("data/store/telemetry.duckdb")

_DDL = """
CREATE TABLE IF NOT EXISTS telemetry (
    ts          TIMESTAMP NOT NULL,
    node        VARCHAR   NOT NULL,
    level       VARCHAR   NOT NULL,
    component   VARCHAR   NOT NULL,
    message     VARCHAR   NOT NULL,
    dialect     VARCHAR   NOT NULL,
    raw_line    VARCHAR
)
"""


def _to_naive_utc(dt: datetime) -> datetime:
    """Strip timezone info after normalising to UTC (DuckDB stores TIMESTAMP as naive)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _to_aware_utc(dt: datetime) -> datetime:
    """Re-attach UTC to a naive datetime read back from DuckDB."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


class TelemetryStore:
    """DuckDB-backed telemetry reservoir with Parquet export."""

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = duckdb.connect(str(db_path))
        self._conn.execute(_DDL)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write_records(self, records: list[TelemetryRecord]) -> None:
        if not records:
            return
        rows = [
            (_to_naive_utc(r.ts), r.node, r.level, r.component, r.message, r.dialect, r.raw_line)
            for r in records
        ]
        self._conn.executemany("INSERT INTO telemetry VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]  # type: ignore[index]

    def time_range(self) -> tuple[datetime, datetime] | None:
        row = self._conn.execute("SELECT MIN(ts), MAX(ts) FROM telemetry").fetchone()
        if row is None or row[0] is None:
            return None
        return (_to_aware_utc(row[0]), _to_aware_utc(row[1]))

    def burst_profile(self, window_s: float) -> tuple[int, datetime] | None:
        """
        Maximum number of WARN/ERROR/FATAL records in any sliding window of
        *window_s* seconds, plus the window-end timestamp where it occurs.

        Window semantics match BurstDetector: the window ends at each alert
        record and extends window_s seconds back (inclusive), over ORIGINAL
        record timestamps. Returns None if the store holds no alert records.
        """
        row = self._conn.execute(
            """
            SELECT a.ts AS window_end, COUNT(*) AS n
            FROM telemetry a
            JOIN telemetry b
              ON b.level IN ('WARN', 'ERROR', 'FATAL')
             AND b.ts <= a.ts
             AND epoch(a.ts) - epoch(b.ts) <= ?
            WHERE a.level IN ('WARN', 'ERROR', 'FATAL')
            GROUP BY a.rowid, a.ts
            ORDER BY n DESC, a.ts ASC
            LIMIT 1
            """,
            [float(window_s)],
        ).fetchone()
        if row is None:
            return None
        return (int(row[1]), _to_aware_utc(row[0]))

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute *sql* (may reference the `telemetry` table) and return rows as dicts."""
        result = self._conn.execute(sql)
        cols = [desc[0] for desc in result.description]  # type: ignore[union-attr]
        return [dict(zip(cols, row, strict=False)) for row in result.fetchall()]

    def all_records(self) -> list[TelemetryRecord]:
        """Return every row as TelemetryRecord, sorted by ts ascending."""
        rows = self._conn.execute(
            "SELECT ts, node, level, component, message, dialect, raw_line "
            "FROM telemetry ORDER BY ts ASC"
        ).fetchall()
        return [
            TelemetryRecord(
                ts=_to_aware_utc(row[0]),
                node=row[1],
                level=row[2],
                component=row[3],
                message=row[4],
                dialect=row[5],
                raw_line=row[6] or "",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_parquet(self, path: Path | None = None) -> Path:
        """Write the full table to a Parquet file and return its path."""
        out = path or self._db_path.with_suffix(".parquet")
        self._conn.execute(f"COPY telemetry TO '{out}' (FORMAT PARQUET)")
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TelemetryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
