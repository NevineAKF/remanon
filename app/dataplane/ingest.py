"""
Ingest CLI — Band A, Layer L5.

Usage:
    python -m app.dataplane.ingest --dataset hdfs_2k
    python -m app.dataplane.ingest --dataset all
    python -m app.dataplane.ingest --dataset hdfs_2k --size full
    python -m app.dataplane.ingest --dataset all --reset

Downloads the raw log file(s) into data/raw/, parses + normalises them
(real Loghub systems only — never synthetic fill), writes the result into
the shared TelemetryStore, then prints per-dataset row counts, the
combined total, and the real combined time span. Every row's `dialect`
column is also its system-of-origin marker (hdfs/bgl/thunderbird/spark/
hadoop) — the one column a multi-system store needs to tell rows apart.

Writes are idempotent per dialect (TelemetryStore.write_records replaces,
never accumulates) — running this any number of times for the same
dataset leaves exactly one copy of it in the store.

--size full pulls the REAL full-size Loghub archive from Zenodo (record
8196385) instead of the ~2000-line GitHub sample, for datasets that have
one registered (hadoop/bgl/hdfs). Systems without a registered full
archive fall back to their 2k sample with a printed note — never
fabricated rows to make up the difference.

If a dataset's URL 404s (or any other download/parse failure), that
dataset is skipped with a clear warning; rows are never fabricated to
compensate.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import httpx
import typer

from app.dataplane.normalizer import HDFSNormalizer, TelemetryRecord, get_normalizer
from app.dataplane.parser import parse_file, read_raw_lines
from app.dataplane.store import TelemetryStore

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Dataset registry — real Loghub samples only, same verified URL pattern:
# raw.githubusercontent.com/logpai/loghub/master/<SYS>/<SYS>_2k.log
#
# full_url (hadoop/bgl/hdfs only) points at the REAL full-size dataset —
# verified live against Zenodo record 8196385 (the loghub dataset
# collection, CC BY 4.0, no access request required):
#   Hadoop.zip -> 978 real per-container .log files, 394,310 real lines
#   BGL.zip    -> one real BGL.log,                 4,747,963 real lines
#   HDFS_v1.zip-> one real HDFS.log,                11,175,629 real lines
# spark_2k / thunderbird_2k have no full_url registered (Thunderbird's
# real archive alone is 211M lines / 2GB compressed) — --size full falls
# back to their 2k sample with a printed note rather than fetching that.
# ---------------------------------------------------------------------------

_LOGHUB_BASE = "https://raw.githubusercontent.com/logpai/loghub/master"
_ZENODO_BASE = "https://zenodo.org/records/8196385/files"

_DATASETS: dict[str, dict] = {
    "hdfs_2k": {
        "url": f"{_LOGHUB_BASE}/HDFS/HDFS_2k.log",
        "filename": "HDFS_2k.log",
        "dialect": "hdfs",
        "full_url": f"{_ZENODO_BASE}/HDFS_v1.zip?download=1",
    },
    "bgl_2k": {
        "url": f"{_LOGHUB_BASE}/BGL/BGL_2k.log",
        "filename": "BGL_2k.log",
        "dialect": "bgl",
        "full_url": f"{_ZENODO_BASE}/BGL.zip?download=1",
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
        "full_url": f"{_ZENODO_BASE}/Hadoop.zip?download=1",
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
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Truncate the whole telemetry table before ingesting — start from a clean store.",
    ),
    size: str = typer.Option(
        "2k",
        "--size",
        help="'2k' (default, GitHub sample) or 'full' (real Zenodo archive — hadoop/bgl/hdfs "
        "only; other datasets fall back to their 2k sample with a printed note).",
    ),
    max_records: int | None = typer.Option(
        None,
        "--max-records",
        help="Cap the number of REAL parsed rows ingested per dataset (default: no cap — "
        "ingest every real row in the file). Never fabricates rows; only ever truncates "
        "a real file's real records.",
    ),
) -> None:
    if size not in ("2k", "full"):
        typer.echo(f"Unknown --size '{size}'. Use '2k' or 'full'.", err=True)
        raise typer.Exit(1)
    if dataset == "all":
        keys = list(_DATASETS)
    elif dataset in _DATASETS:
        keys = [dataset]
    else:
        typer.echo(f"Unknown dataset '{dataset}'. Available: {list(_DATASETS)}, or 'all'", err=True)
        raise typer.Exit(1)

    if reset:
        typer.echo(f"--reset: truncating {_STORE_PATH} …")
        with TelemetryStore(_STORE_PATH) as store:
            store.reset()

    written: dict[str, int] = {}
    for key in keys:
        n = _ingest_one(key, force, size, max_records)
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


def _ingest_one(dataset: str, force: bool, size: str, max_records: int | None) -> int | None:
    """
    Download (2k sample or full Zenodo archive), parse, normalize, and
    store one dataset. Returns the row count written, or None if the
    dataset was skipped (download failure or zero parseable rows) — skips
    are logged, never silently backfilled.
    """
    meta = _DATASETS[dataset]

    raw_path: Path | None
    if size == "full":
        if "full_url" not in meta:
            typer.echo(
                f"[{dataset}] NOTE: no full-size archive registered for this dataset "
                "— falling back to the 2k sample.",
            )
            raw_path = _download_2k(dataset, meta, force)
        else:
            raw_path = _download_and_extract_full(dataset, meta, force)
    else:
        raw_path = _download_2k(dataset, meta, force)

    if raw_path is None:
        return None

    dialect = meta["dialect"]
    records = _parse_dataset(dataset, dialect, raw_path, max_records)
    if not records:
        typer.echo(f"[{dataset}] SKIPPED — no records parsed.", err=True)
        return None

    typer.echo(f"[{dataset}] writing {len(records):,} rows to {_STORE_PATH} …")
    with TelemetryStore(_STORE_PATH) as store:
        store.write_records(records)
    return len(records)


def _download_2k(dataset: str, meta: dict, force: bool) -> Path | None:
    """Download (or reuse the cached) 2k sample file. None on failure."""
    raw_path = _RAW_DIR / meta["filename"]
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        typer.echo(f"[{dataset}] using cached file: {raw_path}")
        return raw_path
    typer.echo(f"[{dataset}] downloading {meta['url']} …")
    try:
        _download(meta["url"], raw_path)
    except httpx.HTTPError as exc:
        typer.echo(f"[{dataset}] SKIPPED — download failed: {exc}", err=True)
        return None
    typer.echo(f"[{dataset}] saved to {raw_path} ({raw_path.stat().st_size:,} bytes)")
    return raw_path


def _download_and_extract_full(dataset: str, meta: dict, force: bool) -> Path | None:
    """
    Download the dataset's full Zenodo archive to a scratch temp dir,
    extract every real *.log entry (sorted by path for a deterministic,
    real concatenation order — some archives, e.g. Hadoop's, are hundreds
    of real per-container log files rather than one flat file; nothing
    here is synthesized, only concatenated), write the result to
    data/raw/<dialect>_full.log, and delete the archive. Cached like the
    2k path: a second run reuses the extracted file unless --force.
    """
    full_filename = _RAW_DIR / f"{meta['dialect']}_full.log"
    if full_filename.exists() and not force:
        typer.echo(f"[{dataset}] using cached full file: {full_filename}")
        return full_filename

    full_url = meta["full_url"]
    typer.echo(f"[{dataset}] downloading FULL archive {full_url} …")
    with tempfile.TemporaryDirectory(prefix="remanon-loghub-") as tmp_dir:
        archive_path = Path(tmp_dir) / f"{dataset}.zip"
        try:
            _download(full_url, archive_path)
        except httpx.HTTPError as exc:
            typer.echo(f"[{dataset}] SKIPPED — full archive download failed: {exc}", err=True)
            return None
        typer.echo(
            f"[{dataset}] downloaded {archive_path.stat().st_size:,} bytes; extracting real log(s) …"
        )

        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zf:
            log_names = sorted(n for n in zf.namelist() if n.endswith(".log"))
            if not log_names:
                typer.echo(
                    f"[{dataset}] SKIPPED — no .log files found inside the archive.", err=True
                )
                return None
            with full_filename.open("wb") as out:
                for name in log_names:
                    with zf.open(name) as entry:
                        shutil.copyfileobj(entry, out)
        # archive_path is deleted here automatically (TemporaryDirectory cleanup).

    typer.echo(
        f"[{dataset}] extracted {len(log_names)} real log file(s) -> "
        f"{full_filename} ({full_filename.stat().st_size:,} bytes)"
    )
    return full_filename


def _parse_dataset(
    dataset: str, dialect: str, raw_path: Path, max_records: int | None
) -> list[TelemetryRecord]:
    """
    HDFS keeps the exact original two-stage pipeline (parse_file →
    HDFSNormalizer). Every other dialect's real line format doesn't fit
    that HDFS-shaped intermediate schema, so its normalizer parses raw
    lines directly via parse_raw() — real regex parsing per dialect, never
    synthetic fill; non-matching lines are skipped and counted.

    max_records caps the REAL rows returned (a real prefix of the real
    file, never a fabricated one) — None means no cap, the whole file.
    """
    if dialect == "hdfs":
        result = parse_file(raw_path)
        typer.echo(
            f"[{dataset}] parsed {len(result.records):,} lines, skipped {result.skipped:,} malformed"
        )
        normalizer = HDFSNormalizer()
        parsed = result.records[:max_records] if max_records is not None else result.records
        records = [normalizer.normalize(p) for p in parsed]
        if max_records is not None and len(parsed) < len(result.records):
            typer.echo(
                f"[{dataset}] capped at --max-records {max_records:,} "
                f"(real file has {len(result.records):,} parseable lines)"
            )
        return records

    normalizer = get_normalizer(dialect)
    lines = read_raw_lines(raw_path)
    records: list[TelemetryRecord] = []
    skipped = 0
    capped = False
    for line in lines:
        if max_records is not None and len(records) >= max_records:
            capped = True
            break
        record = normalizer.parse_raw(line)
        if record is None:
            skipped += 1
        else:
            records.append(record)
    typer.echo(f"[{dataset}] parsed {len(records):,} lines, skipped {skipped:,} malformed")
    if capped:
        typer.echo(f"[{dataset}] capped at --max-records {max_records:,}")
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
