# Remanon — Architecture

> Status: **scaffold / stub** — implementation pending.

## Overview

Remanon is a shared-memory multi-agent runtime designed for AMD MI300X.
Its central insight is that 192 GB of unified HBM3 is large enough to hold
multiple full-parameter models simultaneously, eliminating PCIe data movement
between agents.

---

## Contract Layer (Contract A)

Defines the *interfaces* that all runtime components program against.
No GPU or inference code lives here.

| Symbol | Role |
|--------|------|
| `LeaseManager` | Allocates and releases named HBM3 regions |
| `Materializer` | Loads checkpoints into leased regions |
| `Generator` | Runs inference, writes output artifacts |

Artifact schemas (JSON Schema draft-07) enforce the envelope that each
specialized agent must produce.

---

## Band B — Core Runtime (Memory Arbiter)

Implemented CPU-only; all sizing numbers are configured placeholders pending
the measured budget sheet (**D-03**) from the live MI300X.

| Module | Responsibility |
|--------|----------------|
| `memory_model.py` | Explicit HBM3 model: capacity (192 GB), headroom, per-model weights/master GB, per-agent delta budgets — placeholder values, real arithmetic (D-03) |
| `registry.py` | Artifact catalog + `EngineRegistry` (model → engine base_url/port; health via Contract B `/v1/models`) |
| `materializer.py` | `LazyMaterializer`: one-time master prefill per model through Contract B; idempotent and concurrency-safe (per-model `asyncio.Lock`); counts `prefills_avoided` |
| `residency.py` | `ResidencyManager`: leases pin blocks. **Invariant:** a block with ≥ 1 active lease can never be evicted — `evict()` raises `PinnedBlockError`, and it is the only eviction code path |
| `budgeter.py` | Admission law: `admit ⇔ weights + Σ masters + Σ active_deltas + request ≤ capacity − headroom`. Pressure order: (a) shrink new deltas, (b) reject with `BudgetExceeded`, (c) never touch pinned masters. Byte-exact integer ledger |
| `metrics.py` | Counters/gauges: `prefills_avoided`, `gb_saved_vs_per_agent` (= `(agents_sharing − 1) × master_gb` per model), `active_leases`, ledger snapshot. Plain-dict export for the L9 dashboard |

---

## Band A — Application Layers (L5–L8)

| Layer | Package | Responsibility |
|-------|---------|----------------|
| L8 | `app/orchestrator/` | Deterministic async state machine (**D-02**): `INTAKE → TRIAGE → FAN_OUT → JOIN → REPORTER → EMIT`. Per-worker timeout at JOIN; timeout or twice-invalid output degrades that artifact to `{"status": "inconclusive"}` and the pipeline continues. Append-only `EventLog` (feeds dashboard L9). Burst intake: N WARN/ERROR records in a sliding window opens a case. `run_demo.py` runs the whole loop end-to-end |
| L7 | `app/agents/` | Five agents. `run(case)` = lease (Contract A) → materialize → generate **through the core** → parse (fence-tolerant) → schema-validated `Artifact`. Hunter augments its prompt with SQL evidence from the TelemetryStore |
| L6 | `app/adapter/` | Role prompt templates (`[ROLE:name]` markers, JSON-only output), `DigestBuilder` (compact master context via SQL — materialized and pinned at boot), Contract B client |
| L5 | `app/dataplane/` | Telemetry reservoir (see the Layer L5 section above) |

---

## Contract B — Inference API

Remanon wraps vLLM's OpenAI-compatible REST API as "Contract B".
The `deploy/mock_engine/` FastAPI server imitates this surface for
local development without a GPU.

Endpoints mirrored:
- `POST /v1/chat/completions`
- `GET  /v1/models`

---

## Layer L5 — Data Plane / Telemetry Reservoir

The data plane ingests, normalises, stores, and replays real system log data
so that agents can reason over historic and live telemetry without touching a
GPU.

### Decision D-01 — DuckDB + Parquet

Raw records are persisted in a single DuckDB database file
(`data/store/telemetry.duckdb`).  DuckDB provides SQL over the full
dataset in-process (no server required) and can export Parquet snapshots
on demand for downstream consumers.

```
data/
├── raw/                       ← downloaded log files (git-ignored)
│   └── HDFS_2k.log
└── store/
    ├── telemetry.duckdb       ← primary store (D-01)
    └── telemetry.parquet      ← snapshot export (optional)
```

### Modules

| Module | Responsibility |
|--------|----------------|
| `ingest.py` | CLI: download → parse → normalise → write; `python -m app.dataplane.ingest --dataset hdfs_2k` |
| `parser.py` | Dialect-specific line parser (HDFS format: `YYMMDD HHMMSS pid LEVEL Component: message`). Skips malformed lines; never raises |
| `normalizer.py` | Maps parsed lines to canonical `TelemetryRecord(ts, node, level, component, message, dialect, raw_line)`. Extensible: add dialects by subclassing `BaseNormalizer` |
| `store.py` | `TelemetryStore`: `write_records()`, `query(sql)`, `time_range()`, `count()`, `to_parquet()` |
| `replayer.py` | Async generator `stream()` emitting records in chronological order with configurable `speed_factor` (default 60 → 1 real hour = 1 demo minute) |

### Adding a new log dialect

1. Add a `BaseNormalizer` subclass in `normalizer.py` with `dialect = "my_dialect"`.
2. Register it in the `NORMALIZERS` dict.
3. Add a dataset entry in `ingest.py`'s `_DATASETS` registry.

No other files need changing.

---

## Data Flow (happy path)

```
User Request
    │
    ▼
Orchestrator.triage()
    │  picks agent set
    ▼
[TriageAgent] ──artifact──► Registry
    │
    ├──► [CorrelatorAgent]
    ├──► [HunterAgent]
    ├──► [TopologyAgent]
    └──► [ReporterAgent]
              │
              ▼
         Final Report
```

All inter-agent data passes as HBM3 region handles (zero-copy).

---

## Deployment

See `deploy/docker-compose.yml` for the service graph.
