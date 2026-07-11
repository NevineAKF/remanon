"""
Band B — budgets (Memory Arbiter).

Enforced today and contract-tested: the byte-exact HBM3 admission law
(MemoryBudgeter over a BudgetLedger), pinned-block protection (weights and
masters have no removal API — pressure rule (c)), and per-agent delta caps.
The only stub in this file is the token-budget Budgeter class below.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from contracts.contract_a import AgentName
from core.memory_model import GIB, MemoryModel, gb_to_bytes


@dataclass
class Budget:
    agent: AgentName
    max_tokens: int = 2048
    max_hbm3_bytes: int = 8 * 1024**3  # 8 GiB default
    priority: int = 5  # 1 (highest) – 10 (lowest)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageRecord:
    agent: AgentName
    tokens_used: int = 0
    hbm3_bytes_used: int = 0


class Budgeter:
    """
    Per-agent token-budget bookkeeping (stub — advisory only).

    Records budgets/usage and answers check_token_budget(), but no runtime
    path consults it yet: wiring token enforcement into the generation
    gateway is roadmap work. The HBM3 admission law in this file is NOT part
    of this stub — MemoryBudgeter enforces it today.
    """

    def __init__(self) -> None:
        self._budgets: dict[AgentName, Budget] = {}
        self._usage: dict[AgentName, UsageRecord] = {}

    def set_budget(self, budget: Budget) -> None:
        self._budgets[budget.agent] = budget
        if budget.agent not in self._usage:
            self._usage[budget.agent] = UsageRecord(agent=budget.agent)

    def get_budget(self, agent: AgentName) -> Budget:
        if agent not in self._budgets:
            raise KeyError(f"No budget configured for agent: {agent}")
        return self._budgets[agent]

    def record_usage(self, agent: AgentName, tokens: int, hbm3_bytes: int = 0) -> None:
        if agent not in self._usage:
            self._usage[agent] = UsageRecord(agent=agent)
        rec = self._usage[agent]
        rec.tokens_used += tokens
        rec.hbm3_bytes_used += hbm3_bytes

    def check_token_budget(self, agent: AgentName, requested: int) -> bool:
        """Return True if *requested* tokens fit within remaining budget."""
        budget = self.get_budget(agent)
        used = self._usage.get(agent, UsageRecord(agent=agent)).tokens_used
        return (used + requested) <= budget.max_tokens

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "agent": a,
                "tokens_used": self._usage[a].tokens_used,
                "max_tokens": self._budgets[a].max_tokens,
                "hbm3_bytes_used": self._usage[a].hbm3_bytes_used,
                "max_hbm3_bytes": self._budgets[a].max_hbm3_bytes,
            }
            for a in self._budgets
        ]


# ---------------------------------------------------------------------------
# HBM3 admission law (Phase 3 — Memory Arbiter)
#
#   admit  ⇔  weights + Σ masters + Σ active_deltas + request ≤ capacity − headroom
#
# Pressure resolution order:
#   (a) shrink new deltas down to what fits (callers may queue/retry),
#   (b) reject new admissions with an explicit BudgetExceeded,
#   (c) NEVER touch pinned weights/masters — no eviction API exists here.
#
# All accounting is in exact integer bytes: precision over speed.
# ---------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Admission rejected: the request cannot fit within capacity − headroom."""


@dataclass(frozen=True, slots=True)
class DeltaGrant:
    """Result of a successful delta admission (possibly shrunk under pressure)."""

    key: str
    agent: str
    granted_bytes: int
    requested_bytes: int

    @property
    def granted_gb(self) -> float:
        return self.granted_bytes / GIB

    @property
    def requested_gb(self) -> float:
        return self.requested_bytes / GIB

    @property
    def shrunk(self) -> bool:
        return self.granted_bytes < self.requested_bytes


class BudgetLedger:
    """Live per-partition byte accounting: weights / masters / deltas."""

    def __init__(self, memory_model: MemoryModel) -> None:
        self._capacity_b = memory_model.capacity_bytes
        self._headroom_b = memory_model.headroom_bytes
        self._weights: dict[str, int] = {}  # model → bytes
        self._masters: dict[str, int] = {}  # model → bytes
        self._deltas: dict[str, int] = {}  # grant key → bytes
        self._delta_agents: dict[str, str] = {}  # grant key → agent

    # --- capacity ---

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_b

    @property
    def usable_bytes(self) -> int:
        return self._capacity_b - self._headroom_b

    # --- partitions ---

    @property
    def weights_bytes(self) -> int:
        return sum(self._weights.values())

    @property
    def masters_bytes(self) -> int:
        return sum(self._masters.values())

    @property
    def deltas_bytes(self) -> int:
        return sum(self._deltas.values())

    @property
    def used_bytes(self) -> int:
        return self.weights_bytes + self.masters_bytes + self.deltas_bytes

    @property
    def available_bytes(self) -> int:
        return self.usable_bytes - self.used_bytes

    def agent_delta_bytes(self, agent: str) -> int:
        return sum(b for key, b in self._deltas.items() if self._delta_agents[key] == agent)

    # --- mutators (called under the budgeter lock) ---

    def add_weights(self, model: str, size_bytes: int) -> None:
        self._weights[model] = self._weights.get(model, 0) + size_bytes

    def add_master(self, model: str, size_bytes: int) -> None:
        self._masters[model] = self._masters.get(model, 0) + size_bytes

    def add_delta(self, key: str, agent: str, size_bytes: int) -> None:
        if key in self._deltas:
            raise ValueError(f"Delta grant key already active: {key!r}")
        self._deltas[key] = size_bytes
        self._delta_agents[key] = agent

    def remove_delta(self, key: str) -> int:
        """Free a delta grant; returns bytes freed (0 if key unknown — safe as a release hook)."""
        self._delta_agents.pop(key, None)
        return self._deltas.pop(key, 0)

    # --- export ---

    def snapshot(self) -> dict[str, Any]:
        per_agent: dict[str, float] = {}
        for key, b in self._deltas.items():
            agent = self._delta_agents[key]
            per_agent[agent] = per_agent.get(agent, 0.0) + b / GIB
        return {
            "capacity_gb": self._capacity_b / GIB,
            "headroom_gb": self._headroom_b / GIB,
            "usable_gb": self.usable_bytes / GIB,
            "weights_gb": self.weights_bytes / GIB,
            "masters_gb": self.masters_bytes / GIB,
            "deltas_gb": self.deltas_bytes / GIB,
            "used_gb": self.used_bytes / GIB,
            "available_gb": self.available_bytes / GIB,
            "available_bytes": self.available_bytes,
            "per_agent_delta_gb": per_agent,
            "active_delta_grants": len(self._deltas),
        }


class MemoryBudgeter:
    """
    The admission arbiter. Enforces the admission law over a BudgetLedger.

    Weights and masters are pinned at registration and cannot be released:
    this class deliberately has no API to remove them (pressure rule (c)).
    """

    def __init__(self, memory_model: MemoryModel, ledger: BudgetLedger | None = None) -> None:
        self._mm = memory_model
        self.ledger = ledger if ledger is not None else BudgetLedger(memory_model)
        self._lock = asyncio.Lock()

    async def pin_model(self, model: str) -> None:
        """Admit and pin a model's weights + master block. Raises BudgetExceeded if it cannot fit."""
        spec = self._mm.models[model]
        weights_b = gb_to_bytes(spec.weights_gb)
        master_b = gb_to_bytes(spec.master_gb)
        async with self._lock:
            needed = weights_b + master_b
            if self.ledger.used_bytes + needed > self.ledger.usable_bytes:
                raise BudgetExceeded(
                    f"Cannot pin {model!r}: needs {needed} B, "
                    f"only {self.ledger.available_bytes} B available "
                    f"(usable={self.ledger.usable_bytes} B)"
                )
            self.ledger.add_weights(model, weights_b)
            self.ledger.add_master(model, master_b)

    async def admit_delta(
        self,
        key: str,
        agent: str,
        request_gb: float,
        *,
        min_gb: float | None = None,
    ) -> DeltaGrant:
        """
        Admit a working delta under the admission law.

        If the full request does not fit but *min_gb* does, the grant is
        shrunk to what fits (pressure rule (a)). Otherwise BudgetExceeded is
        raised (rule (b)). Pinned weights/masters are never candidates (rule (c)).
        """
        request_b = gb_to_bytes(request_gb)
        min_b = gb_to_bytes(min_gb) if min_gb is not None else request_b
        async with self._lock:
            grantable = self.ledger.available_bytes
            agent_cap_gb = self._mm.agent_delta_budget_gb.get(agent)
            if agent_cap_gb is not None:
                agent_remaining = gb_to_bytes(agent_cap_gb) - self.ledger.agent_delta_bytes(agent)
                grantable = min(grantable, agent_remaining)

            if request_b <= grantable:
                granted_b = request_b
            elif min_b <= grantable:
                granted_b = grantable  # pressure (a): shrink to fit
            else:
                raise BudgetExceeded(
                    f"Delta for agent {agent!r} rejected: requested {request_b} B "
                    f"(min {min_b} B), grantable {grantable} B"
                )
            self.ledger.add_delta(key, agent, granted_b)
            return DeltaGrant(
                key=key, agent=agent, granted_bytes=granted_b, requested_bytes=request_b
            )

    def release_delta(self, key: str) -> int:
        """Free the delta grant *key*; returns bytes freed (0 if unknown)."""
        return self.ledger.remove_delta(key)
