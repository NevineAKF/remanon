# Remanon

**GPU-resident shared-memory runtime for multi-agent AI on AMD MI300X.**

Remanon keeps model weights, KV caches, and intermediate artifacts in HBM3
so that a coordinated swarm of specialized agents can share data without PCIe
round-trips.

---

## Three-Band Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║  BAND A — CONTRACT LAYER  (interfaces, schemas, leases)                 ║
║                                                                          ║
║   contract_a.py          artifact_schemas/                               ║
║   ├─ LeaseManager        ├─ triage.json                                  ║
║   ├─ Materializer        ├─ correlator.json                              ║
║   └─ Generator           ├─ hunter.json                                  ║
║                          ├─ topology.json                                ║
║                          └─ reporter.json                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  BAND B — CORE RUNTIME   (registry, residency, budget)                  ║
║                                                                          ║
║   core/                                                                  ║
║   ├─ registry.py         — artifact + agent registry                     ║
║   ├─ materializer.py     — HBM3 tensor materialisation                   ║
║   ├─ residency.py        — GPU-memory residency tracking                 ║
║   └─ budgeter.py         — per-agent token / memory budgets              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  BAND C — APPLICATION    (orchestrator, agents, adapter, dataplane)     ║
║                                                                          ║
║   app/                                                                   ║
║   ├─ orchestrator/       — triage → route → collect                      ║
║   ├─ agents/             — triage, correlator, hunter, topology,         ║
║   │                        reporter                                      ║
║   ├─ adapter/            — Contract B ↔ OpenAI-compatible vLLM shim      ║
║   └─ dataplane/          — zero-copy tensor passing between agents       ║
╚══════════════════════════════════════════════════════════════════════════╝

  dashboard/   — observability UI (Prometheus + Rich TUI)
  deploy/      — Docker Compose + mock_engine (Contract B stub)
```

---

## Quick Start (CPU / no GPU required)

```bash
# 1. create venv
uv venv
uv pip install -e ".[dev]"

# 2. start mock inference engine (imitates vLLM OpenAI API)
cd deploy/mock_engine
uvicorn main:app --port 8000

# 3. run tests
pytest

# 4. lint
ruff check .
```

---

## Project Layout

```
remanon/
├── contracts/
│   ├── contract_a.py          # Band A interface definitions
│   └── artifact_schemas/      # JSON Schema per agent output
├── core/                      # Band B runtime stubs
├── app/                       # Band C application layer
│   ├── orchestrator/
│   ├── agents/
│   ├── adapter/
│   └── dataplane/
├── dashboard/                 # Observability placeholder
├── deploy/
│   ├── docker-compose.yml
│   └── mock_engine/           # FastAPI Contract B mock
├── docs/
│   └── ARCHITECTURE.md
└── tests/
```

---

## Live demo

`dashboard/showcase/index.html` is a self-contained, static build of the
Live Operations Theater dashboard — a real recorded run (real HDFS
production logs, replayed through the full agent pipeline), not a mock-up.
It embeds its data inline, so it needs no server: open it directly, or host
it anywhere static files are served, e.g. GitHub Pages at:

```
https://nevineakf.github.io/remanon/
```

(once Pages is enabled for this repo, pointed at `dashboard/showcase/`).

**Regenerate the recording** after a pipeline or dashboard change:

```bash
python -m app.orchestrator.run_demo --record --no-export --speed 1000
```

This replays the full telemetry store, captures the complete EventLog
stream plus a series of timestamped `/api/state` snapshots, and writes:

- `dashboard/showcase/run_recording.json` — the recording, as data
- `dashboard/showcase/index.html` — the same page with that recording
  spliced into its inline `<script id="run-recording-data">` data island

Open `dashboard/showcase/index.html` (via `file://` or a static host) and
press **▶ RUN LIVE DEMO** to replay it. The footer always discloses that
it's a recorded replay with a mock inference engine — it never claims live.

---

## Hardware Target

| Component | Spec |
|-----------|------|
| GPU | AMD Instinct MI300X |
| HBM3 | 192 GB unified memory |
| Interconnect | AMD Infinity Fabric |
| Host | ROCm 6.x, Python 3.12 |

> **Note:** All logic stubs run on CPU. GPU paths are gated behind
> `REMANON_GPU=1` (not yet implemented).
