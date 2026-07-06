"""
Band A, Layer L8 — orchestrator (decision D-02).

Deterministic async state machine per case:
    INTAKE → TRIAGE → FAN_OUT → JOIN → REPORTER → EMIT

FAN_OUT runs the routed workers concurrently (asyncio.gather semantics via
tasks). JOIN applies a per-worker timeout; on timeout or twice-invalid
output the worker's artifact degrades to {"status": "inconclusive"} and the
pipeline continues. Every state transition and artifact is recorded in an
append-only EventLog (feeds dashboard L9 later).

Intake rule: a case opens when N WARN/ERROR records land within a sliding
time window (log time, not wall time).
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from app.agents.base import BaseAgent
from app.dataplane.normalizer import TelemetryRecord
from contracts.contract_a import Artifact

WORKER_AGENTS = ("correlator", "hunter", "topology")

_ALERT_LEVELS = {"WARN", "ERROR", "FATAL"}


class CaseState(StrEnum):
    INTAKE = "INTAKE"
    TRIAGE = "TRIAGE"
    FAN_OUT = "FAN_OUT"
    JOIN = "JOIN"
    REPORTER = "REPORTER"
    EMIT = "EMIT"


# ---------------------------------------------------------------------------
# Append-only event log
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    ts: datetime
    case_id: str
    kind: (
        str  # "state" | "artifact" | "worker_start" | "worker_end" | "degraded" | "attempt_failed"
    )
    data: dict[str, Any]


class EventLog:
    """Append-only: no mutation or deletion API; reads return copies."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(self, case_id: str, kind: str, **data: Any) -> Event:
        event = Event(ts=datetime.now(UTC), case_id=case_id, kind=kind, data=data)
        self._events.append(event)
        return event

    def events(self, case_id: str | None = None, kind: str | None = None) -> list[Event]:
        selected = self._events
        if case_id is not None:
            selected = [e for e in selected if e.case_id == case_id]
        if kind is not None:
            selected = [e for e in selected if e.kind == kind]
        return list(selected)

    def __len__(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Burst-based intake
# ---------------------------------------------------------------------------


class BurstDetector:
    """Opens a case when >= threshold WARN/ERROR records fall in a sliding window."""

    def __init__(self, threshold: int = 5, window_s: float = 60.0) -> None:
        self._threshold = threshold
        self._window = timedelta(seconds=window_s)
        self._hits: deque[TelemetryRecord] = deque()

    def observe(self, record: TelemetryRecord) -> dict[str, Any] | None:
        if record.level not in _ALERT_LEVELS:
            return None
        self._hits.append(record)
        cutoff = record.ts - self._window
        while self._hits and self._hits[0].ts < cutoff:
            self._hits.popleft()
        if len(self._hits) < self._threshold:
            return None
        triggers = list(self._hits)
        self._hits.clear()
        return {
            "case_id": uuid.uuid4().hex,
            "opened_at": record.ts.isoformat(),
            "record_count": len(triggers),
            "trigger_records": [
                {
                    "ts": r.ts.isoformat(),
                    "node": r.node,
                    "level": r.level,
                    "component": r.component,
                    "message": r.message[:200],
                }
                for r in triggers
            ],
        }


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


def _degraded(agent: str, reason: str) -> dict[str, Any]:
    return {"status": "inconclusive", "agent": agent, "reason": reason}


class Orchestrator:
    """Runs one case through the deterministic D-02 pipeline."""

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        event_log: EventLog | None = None,
        *,
        worker_timeout_s: float = 10.0,
        max_attempts: int = 2,
    ) -> None:
        self._agents = agents
        self.event_log = event_log if event_log is not None else EventLog()
        self._worker_timeout_s = worker_timeout_s
        self._max_attempts = max_attempts

    async def run_case(self, case: dict[str, Any]) -> Artifact | dict[str, Any]:
        case_id = case["case_id"]
        self._log_state(case_id, CaseState.INTAKE)

        # --- TRIAGE ---
        self._log_state(case_id, CaseState.TRIAGE)
        triage_result = await self._run_with_retry(case_id, "triage", case)
        if isinstance(triage_result, Artifact):
            self._log_artifact(case_id, triage_result)
            triage_payload: dict[str, Any] = triage_result.payload
            routing = [
                name
                for name in triage_payload["routing"]
                if name in WORKER_AGENTS and name in self._agents
            ]
        else:
            triage_payload = triage_result  # degraded → route to every worker
            routing = [name for name in WORKER_AGENTS if name in self._agents]

        # --- FAN_OUT: workers start concurrently ---
        self._log_state(case_id, CaseState.FAN_OUT)
        tasks: dict[str, asyncio.Task] = {
            name: asyncio.create_task(self._timed_worker(case_id, name, case)) for name in routing
        }

        # --- JOIN: per-worker timeout; degrade and continue ---
        self._log_state(case_id, CaseState.JOIN)
        worker_results: dict[str, Artifact | dict[str, Any]] = {}
        for name, task in tasks.items():
            try:
                result = await asyncio.wait_for(task, timeout=self._worker_timeout_s)
            except TimeoutError:
                result = _degraded(name, "timeout")
                self.event_log.append(case_id, "degraded", agent=name, reason="timeout")
            if isinstance(result, Artifact):
                self._log_artifact(case_id, result)
            worker_results[name] = result

        # --- REPORTER ---
        self._log_state(case_id, CaseState.REPORTER)
        reporter_case = {
            **case,
            "triage": triage_payload,
            "workers": {
                name: (res.payload if isinstance(res, Artifact) else res)
                for name, res in worker_results.items()
            },
        }
        final = await self._run_with_retry(case_id, "reporter", reporter_case)
        if isinstance(final, Artifact):
            self._log_artifact(case_id, final)

        # --- EMIT ---
        self._log_state(case_id, CaseState.EMIT)
        return final

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _timed_worker(
        self, case_id: str, agent_name: str, case: dict[str, Any]
    ) -> Artifact | dict[str, Any]:
        self.event_log.append(case_id, "worker_start", agent=agent_name)
        try:
            return await self._run_with_retry(case_id, agent_name, case)
        finally:
            self.event_log.append(case_id, "worker_end", agent=agent_name)

    async def _run_with_retry(
        self, case_id: str, agent_name: str, case: dict[str, Any]
    ) -> Artifact | dict[str, Any]:
        """Up to max_attempts; on final failure return a degraded artifact."""
        agent = self._agents[agent_name]
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await agent.run(case)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                self.event_log.append(
                    case_id,
                    "attempt_failed",
                    agent=agent_name,
                    attempt=attempt,
                    error=str(exc),
                )
        self.event_log.append(case_id, "degraded", agent=agent_name, reason="invalid_output")
        return _degraded(agent_name, f"invalid_output: {last_error}")

    def _log_state(self, case_id: str, state: CaseState) -> None:
        self.event_log.append(case_id, "state", state=state.value)

    def _log_artifact(self, case_id: str, artifact: Artifact) -> None:
        self.event_log.append(
            case_id, "artifact", agent=artifact.agent, artifact_raw=artifact.to_dict()
        )
