FROM python:3.12-slim

WORKDIR /app

# Full repo first (app/core/contracts + dashboard/ + deploy/ as namespace
# packages, README.md for hatchling's `readme` field) — the project layout
# only has three declared wheel packages (contracts/core/app), so `pip
# install .` alone would not ship dashboard/ or deploy/; running via
# `python -m` with WORKDIR=/app as cwd resolves all of them straight from
# source instead, matching local `uv run` dev behaviour exactly.
COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8080

# Real data only, never synthetic: ingest is deferred to container start
# (not RUN, at build time) so a network-restricted `docker build` still
# succeeds — the five real Loghub samples are small and download in
# seconds on `docker run`. Idempotent (app/dataplane/store.py replaces by
# dialect), so re-running the container never accumulates duplicate rows.
# --dashboard-host 0.0.0.0 is required for the dashboard to be reachable
# through the container's port mapping (uvicorn's own default is
# loopback-only, which `-p 8080:8080` cannot reach).
CMD ["sh", "-c", "python -m app.dataplane.ingest --dataset all && python -m app.orchestrator.run_demo --speed 1000 --dashboard --dashboard-host 0.0.0.0"]
