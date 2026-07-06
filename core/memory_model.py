"""
Explicit HBM3 memory model — Band B.

ALL numeric values in this module are CONFIGURED PLACEHOLDERS pending the
measured budget sheet (D-03) from the live MI300X. The arithmetic is real;
only the numbers get replaced once measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024**3


def gb_to_bytes(gb: float) -> int:
    """Convert GB (GiB) to an exact integer byte count."""
    return round(gb * GIB)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Per-model memory footprint. Placeholder values pending D-03."""

    name: str
    weights_gb: float
    master_gb: float  # shared master context (prefill) block


# Dev engine topology: 4 engines serve 5 agents (reporter shares the
# correlator-13b engine). Placeholder pending D-03.
DEFAULT_MODELS: tuple[str, ...] = (
    "remanon-triage-7b",
    "remanon-correlator-13b",
    "remanon-hunter-13b",
    "remanon-topology-7b",
)

AGENT_MODEL_MAP: dict[str, str] = {
    "triage": "remanon-triage-7b",
    "correlator": "remanon-correlator-13b",
    "hunter": "remanon-hunter-13b",
    "topology": "remanon-topology-7b",
    "reporter": "remanon-correlator-13b",  # shares the 13b engine (placeholder, D-03)
}


def _default_model_specs() -> dict[str, ModelSpec]:
    # Placeholder weights/master sizes pending D-03.
    return {
        "remanon-triage-7b": ModelSpec("remanon-triage-7b", weights_gb=14.0, master_gb=6.0),
        "remanon-correlator-13b": ModelSpec(
            "remanon-correlator-13b", weights_gb=26.0, master_gb=10.0
        ),
        "remanon-hunter-13b": ModelSpec("remanon-hunter-13b", weights_gb=26.0, master_gb=10.0),
        "remanon-topology-7b": ModelSpec("remanon-topology-7b", weights_gb=14.0, master_gb=6.0),
    }


def _default_delta_budgets() -> dict[str, float]:
    # Per-agent working-delta budgets. Placeholder pending D-03.
    return {
        "triage": 4.0,
        "correlator": 6.0,
        "hunter": 6.0,
        "topology": 4.0,
        "reporter": 6.0,
    }


@dataclass(frozen=True, slots=True)
class MemoryModel:
    """
    The single source of truth for HBM3 capacity arithmetic.

    total_capacity_gb: physical HBM3 on the MI300X (placeholder, D-03).
    headroom_gb:       reserved slack never handed to any partition.
    models:            per-model weights + master block sizes.
    agent_delta_budget_gb: per-agent cap on working deltas; agents absent
                           from this mapping have no per-agent cap.
    """

    total_capacity_gb: float = 192.0  # MI300X HBM3 — placeholder pending D-03
    headroom_gb: float = 12.0  # placeholder pending D-03
    models: dict[str, ModelSpec] = field(default_factory=_default_model_specs)
    agent_delta_budget_gb: dict[str, float] = field(default_factory=_default_delta_budgets)

    @property
    def usable_gb(self) -> float:
        return self.total_capacity_gb - self.headroom_gb

    @property
    def capacity_bytes(self) -> int:
        return gb_to_bytes(self.total_capacity_gb)

    @property
    def headroom_bytes(self) -> int:
        return gb_to_bytes(self.headroom_gb)

    @property
    def usable_bytes(self) -> int:
        return self.capacity_bytes - self.headroom_bytes
