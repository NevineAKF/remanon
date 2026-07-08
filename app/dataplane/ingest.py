"""
Ingest CLI — Band A, Layer L5.

Usage:
    python -m app.dataplane.ingest --dataset hdfs_2k
    python -m app.dataplane.ingest --dataset all

Downloads the raw log file(s) into data/raw/, parses + normalises them
(real Loghub systems only — never synthetic fill), writes the result into
the shared TelemetryStore, then prints per-dataset row counts, the
combined total, and the real combined time span. Every row's `dialect`
column is also its system-of-origin marker (hdfs/bgl/thunderbird/spark/
hadoop) — the one column a multi-system store needs to tell rows apart.

If a dataset's URL 404s (or any other download/parse failure), that
dataset is skipped with a clear warning; rows are never fabricated to
compensate.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from app.dataplane.normalizer import HDFSNormalizer, get_normalizer
from app.dataplane.parser import parse_file, read_raw_lines
from app.dataplane.store import TelemetryStore

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Dataset registry — real Loghub samples only, same verified URL pattern:
# raw.githubusercontent.com/logpai/loghub/master/<SYS>/<SYS>_2k.log
# ---------------------------------------------------------------------------

_LOGHUB_BASE = "https://raw.githubusercontent.com/logpai/loghub/master"

_DATASETS: dict[str, dict] = {
    "hdfs_2k": {
        "url": f"{_LOGHUB_BASE}/HDFS/HDFS_2k.log",
        "filename": "HDFS_2k.log",
        "dialect": "hdfs",
    },
    "bgl_2k": {
        "url": f"{_LOGHUB_BASE}/BGL/BGL_2k.log",
        "filename": "BGL_2k.log",
        "dialect": "bgl",
    },
    "thunderbird_2k": {
        "url": f"{_LOGHUB_BASE}/Thunderbird/Thunderbird_2k.log",
        "filename": "Thunderbird_2k.log",
        "dialect": "thunderbird",
    },
    "spark_2k": {
        "url": f"{_LOGHUB_BASE}/Spark/Spark_2k.log",
        "filename": "Spark_2k.log",
        "dialect": "spark",
    },
    "hadoop_2k": {
        "url": f"{_LOGHUB_BASE}/Hadoop/Hadoop_2k.log",
        "filename": "Hadoop_2k.log",
        "dialect": "hadoop",
    },
}

_RAW_DIR = Path("data/raw")
_STORE_PATH = Path("data/store/telemetry.duckdb")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    dataset: str = typer.Option(
        "hdfs_2k", help="Dataset key from the registry, or 'all' to ingest every one."
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if local file exists."),
) -> None:
    if dataset == "all":
        keys = list(_DATASETS)
    elif dataset in _DATASETS:
        keys = [dataset]
    else:
        typer.echo(f"Unknown dataset '{dataset}'. Available: {list(_DATASETS)}, or 'all'", err=True)
        raise typer.Exit(1)

    written: dict[str, int] = {}
    for key in keys:
        n = _ingest_one(key, force)
        if n is not None:
            written[key] = n

    if not written:
        typer.echo("\nNo datasets ingested — nothing written.", err=True)
        raise typer.Exit(1)

    if len(keys) > 1:
        typer.echo("\n--- rows written this run, per dataset ---")
        for key in keys:
            status = f"{written[key]:,}" if key in written else "SKIPPED"
            typer.echo(f"  {key:16s}: {status}")

    with TelemetryStore(_STORE_PATH) as store:
        total = store.count()
        span = store.time_range()

    typer.echo(f"\nCombined store total : {total:,} rows")
    if span:
        typer.echo(f"Combined time span   : {span[0].isoformat()} → {span[1].isoformat()}")


# ---------------------------------------------------------------------------
# Per-dataset ingest
# ---------------------------------------------------------------------------


def _ingest_one(dataset: str, force: bool) -> int | None:
    """
    Download, parse, normalize, and store one dataset. Returns the row
    count written, or None if the dataset was skipped (download failure or
    zero parseable rows) — skips are logged, never silently backfilled.
    """
    meta = _DATASETS[dataset]
    raw_path = _RAW_DIR / meta["filename"]

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        typer.echo(f"[{dataset}] using cached file: {raw_path}")
    else:
        typer.echo(f"[{dataset}] downloading {meta['url']} …")
        try:
            _download(meta["url"], raw_path)
        except httpx.HTTPError as exc:
            typer.echo(f"[{dataset}] SKIPPED — download failed: {exc}", err=True)
            return None
        typer.echo(f"[{dataset}] saved to {raw_path} ({raw_path.stat().st_size:,} bytes)")

    dialect = meta["dialect"]
    records = _parse_dataset(dataset, dialect, raw_path)
    if not records:
        typer.echo(f"[{dataset}] SKIPPED — no records parsed.", err=True)
        return None

    typer.echo(f"[{dataset}] writing {len(records):,} rows to {_STORE_PATH} …")
    with TelemetryStore(_STORE_PATH) as store:
        store.write_records(records)
    return len(records)


def _parse_dataset(dataset: str, dialect: str, raw_path: Path) -> list:
    """
    HDFS keeps the exact original two-stage pipeline (parse_file →
    HDFSNormalizer). Every other dialect's real line format doesn't fit
    that HDFS-shaped intermediate schema, so its normalizer parses raw
    lines directly via parse_raw() — real regex parsing per dialect, never
    synthetic fill; non-matching lines are skipped and counted.
    """
    if dialect == "hdfs":
        result = parse_file(raw_path)
        typer.echo(
            f"[{dataset}] parsed {len(result.records):,} lines, skipped {result.skipped:,} malformed"
        )
        normalizer = HDFSNormalizer()
        return [normalizer.normalize(p) for p in result.records]

    normalizer = get_normalizer(dialect)
    lines = read_raw_lines(raw_path)
    records = []
    skipped = 0
    for line in lines:
        record = normalizer.parse_raw(line)
        if record is None:
            skipped += 1
        else:
            records.append(record)
    typer.echo(f"[{dataset}] parsed {len(records):,} lines, skipped {skipped:,} malformed")
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=65536):
                fh.write(chunk)


# ---------------------------------------------------------------------------
# Entry point (python -m app.dataplane.ingest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
