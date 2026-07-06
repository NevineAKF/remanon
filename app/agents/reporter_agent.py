"""Band A, Layer L7 — Reporter agent: consolidates triage + worker artifacts."""

from __future__ import annotations

from app.agents.base import BaseAgent


class ReporterAgent(BaseAgent):
    name = "reporter"
