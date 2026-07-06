"""
Ingest CLI — Band A, Layer L5.

Usage:
    python -m app.dataplane.ingest --dataset hdfs_2k

Downloads the raw log file into data/raw/, parses + normalises it, writes
the result into TelemetryStore, then prints row count and time range.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from app.dataplane.normalizer import HDFSNormalizer
from app.dataplane.parser import parse_file
from app.dataplane.store import TelemetryStore

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_DATASETS: dict[str, dict] = {
    "hdfs_2k": {
        "url": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
        "filename": "HDFS_2k.log",
        "dialect": "hdfs",
    },
}

_RAW_DIR = Path("data/raw")
_STORE_PATH = Path("data/store/telemetry.duckdb")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    dataset: str = typer.Option("hdfs_2k", help="Dataset key from the built-in registry."),
    force: bool = typer.Option(False, "--force", help="Re-download even if local file exists."),
) -> None:
    if dataset not in _DATASETS:
        typer.echo(f"Unknown dataset '{dataset}'. Available: {list(_DATASETS)}", err=True)
        raise typer.Exit(1)

    meta = _DATASETS[dataset]
    raw_path = _RAW_DIR / meta["filename"]

    # --- download ---
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        typer.echo(f"Using cached file: {raw_path}")
    else:
        typer.echo(f"Downloading {meta['url']} …")
        _download(meta["url"], raw_path)
        typer.echo(f"Saved to {raw_path} ({raw_path.stat().st_size:,} bytes)")

    # --- parse ---
    typer.echo("Parsing …")
    result = parse_file(raw_path)
    typer.echo(f"  Parsed {len(result.records):,} lines, skipped {result.skipped:,} malformed")

    if not result.records:
        typer.echo("No records to ingest.", err=True)
        raise typer.Exit(1)

    # --- normalise ---
    normalizer = HDFSNormalizer()
    records = [normalizer.normalize(p) for p in result.records]

    # --- store ---
    typer.echo(f"Writing to {_STORE_PATH} …")
    with TelemetryStore(_STORE_PATH) as store:
        store.write_records(records)
        count = store.count()
        span = store.time_range()

    typer.echo(f"  Total rows in store : {count:,}")
    if span:
        typer.echo(f"  Time range          : {span[0].isoformat()} → {span[1].isoformat()}")


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
