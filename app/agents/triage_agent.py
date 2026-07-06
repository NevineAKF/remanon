"""Band C — Triage agent stub."""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from contracts.contract_a import Artifact


class TriageAgent(BaseAgent):
    name = "triage"

    async def run(self, context: dict[str, Any]) -> Artifact:
        raise NotImplementedError("TriageAgent.run — stub")
