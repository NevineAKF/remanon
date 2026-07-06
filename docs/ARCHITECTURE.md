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
