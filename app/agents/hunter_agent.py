"""Band A, Layer L7 — Hunter agent: augments its prompt with SQL evidence."""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent

_EVIDENCE_SQL = (
    "SELECT ts, node, level, component, message FROM telemetry "
    "WHERE level IN ('WARN', 'ERROR', 'FATAL') ORDER BY ts LIMIT 20"
)


class HunterAgent(BaseAgent):
    name = "hunter"

    def user_prompt(self, case: dict[str, Any]) -> str:
        prompt = super().user_prompt(case)
        if self._store is None:
            return prompt
        rows = self._store.query(_EVIDENCE_SQL)
        evidence = "\n".join(
            f"{row['ts']} {row['node']} {row['level']} {row['component']}: {row['message']}"
            for row in rows
        )
        return f"{prompt}\nSQL EVIDENCE (WARN/ERROR):\n{evidence}"
