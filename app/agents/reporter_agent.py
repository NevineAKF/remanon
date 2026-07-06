"""Band C — Reporter agent stub."""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from contracts.contract_a import Artifact


class ReporterAgent(BaseAgent):
    name = "reporter"

    async def run(self, context: dict[str, Any]) -> Artifact:
        raise NotImplementedError("ReporterAgent.run — stub")
