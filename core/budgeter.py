"""Band B — per-agent token / memory budgets (stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.contract_a import AgentName


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
    """Enforces per-agent resource quotas (stub — no enforcement yet)."""

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
