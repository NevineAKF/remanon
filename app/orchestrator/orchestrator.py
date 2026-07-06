"""Band C — task router stub.

Receives an incoming request, calls triage, then fans out to the agent
set specified in the triage artifact's routing list.
"""

from __future__ import annotations

from typing import Any

from core.registry import Registry


class Orchestrator:
    """Routes tasks through the multi-agent pipeline (stub)."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the full pipeline for *task*.

        Steps (stub):
          1. triage
          2. fan-out to routed agents
          3. reporter
          4. return final artifact dict

        All steps raise NotImplementedError until implemented.
        """
        raise NotImplementedError("Orchestrator.run — stub")
