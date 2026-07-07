"""
Core metrics — Band B (Memory Arbiter).

Plain-dict export of counters and gauges; the L9 dashboard consumes this
later. gb_saved_vs_per_agent quantifies the shared-master win:
for each model, (num_agents_sharing − 1) × master_gb, where sharing is
CUMULATIVE (agents that ever leased the master) — releasing a lease does
not undo a saving that already happened.
"""

from __future__ import annotations

from typing import Any

from core.budgeter import MemoryBudgeter
from core.materializer import LazyMaterializer
from core.memory_model import MemoryModel
from core.residency import ResidencyManager


class CoreMetrics:
    def __init__(
        self,
        *,
        memory_model: MemoryModel,
        materializer: LazyMaterializer,
        residency: ResidencyManager,
        budgeter: MemoryBudgeter,
    ) -> None:
        self._mm = memory_model
        self._materializer = materializer
        self._residency = residency
        self._budgeter = budgeter

    def gb_saved_vs_per_agent(self) -> float:
        """HBM3 saved by sharing masters instead of one copy per agent (cumulative)."""
        saved = 0.0
        for model, spec in self._mm.models.items():
            sharing = len(self._residency.agents_ever_leased(model))
            if sharing >= 1:
                saved += (sharing - 1) * spec.master_gb
        return saved

    def export(self) -> dict[str, Any]:
        return {
            "prefills_avoided": self._materializer.prefills_avoided,
            "prefills_performed": self._materializer.prefills_performed,
            "active_leases": self._residency.active_leases,
            "gb_saved_vs_per_agent": self.gb_saved_vs_per_agent(),
            "ledger": self._budgeter.ledger.snapshot(),
        }
