# Remanon — Architecture

> Status: **scaffold / stub** — implementation pending.

## Overview

Remanon is a shared-memory multi-agent runtime designed for AMD MI300X.
Its central insight is that 192 GB of unified HBM3 is large enough to hold
multiple full-parameter models simultaneously, eliminating PCIe data movement
between agents.

---

## Band A — Contract Layer

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

## Band B — Core Runtime

Implements the interfaces declared in Band A.

| Module | Responsibility |
|--------|----------------|
| `registry.py` | Central catalog of live artifacts and agent handles |
| `materializer.py` | Checkpoint loading, tensor layout, dtype promotion |
| `residency.py` | Tracks which tensors reside in HBM3 vs host RAM |
| `budgeter.py` | Per-agent token budgets, memory quotas, priority queues |

---

## Band C — Application Layer

| Package | Responsibility |
|---------|----------------|
| `orchestrator/` | Receives tasks, routes to agents, aggregates results |
| `agents/` | Five domain agents (triage, correlator, hunter, topology, reporter) |
| `adapter/` | Translates OpenAI-compatible vLLM API (Contract B) to internal calls |
| `dataplane/` | Zero-copy tensor references passed between agents via HBM3 handles |

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
