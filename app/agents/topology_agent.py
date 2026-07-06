"""Band C — Topology agent stub."""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from contracts.contract_a import Artifact


class TopologyAgent(BaseAgent):
    name = "topology"

    async def run(self, context: dict[str, Any]) -> Artifact:
        raise NotImplementedError("TopologyAgent.run — stub")
