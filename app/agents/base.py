"""Band C — base class shared by all domain agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from contracts.contract_a import AgentName, Artifact


class BaseAgent(ABC):
    name: AgentName = ""

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> Artifact:
        """Execute the agent and return a validated artifact."""
