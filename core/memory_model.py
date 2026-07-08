"""
Explicit HBM3 memory model — Band B.

ALL numeric values in this module are CONFIGURED PLACEHOLDERS pending
direct MI300X measurement. The arithmetic is real; only the numbers get
replaced once measured. See docs/evidence/D03_budget_sheet.md — Tier 1
grounds the model-load size (gpt-oss-20b: 14.0 GB placeholder here vs.
14.3 GiB measured, 2.1% agreement) and the pinned-residency mechanism
itself (5.8x-32x prefix-reuse speedup, measured) in real AMD hardware;
Tier 2 is this module's own numbers run through the admission-law
arithmetic for the 192 GB MI300X target. Weights/masters/deltas for the
120B/70B/32B-class models and the MI300X capacity itself remain COMPUTED,
not yet MEASURED — that gap is D-03's open follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024**3


def gb_to_bytes(gb: float) -> int:
    """Convert GB (GiB) to an exact integer byte count."""
    return round(gb * GIB)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Per-model memory footprint. See docs/evidence/D03_budget_sheet.md
    Tier 2 for the per-model table these values feed; only the
    gpt-oss-20b-class weights are Tier-1 measured-agreement so far."""

    name: str
    weights_gb: float
    master_gb: float  # shared master context (prefill) block

    def __post_init__(self) -> None:
        # A zero-sized master would silently null the gb_saved metric; the
        # placeholders must be explicit non-zero values (pending D-03).
        if self.weights_gb <= 0 or self.master_gb <= 0:
            raise ValueError(
                f"ModelSpec {self.name!r}: weights_gb and master_gb must be > 0 "
                f"(got weights_gb={self.weights_gb}, master_gb={self.master_gb})"
            )


# Dev engine topology: 4 engines serve 5 agents (reporter shares the
# correlator-13b engine). Display names → real target checkpoints:
# triage=gpt-oss-20b, correlator/reporter=gpt-oss-120b, hunter=llama-3.3-70b,
# topology=qwen3-32b — see docs/evidence/D03_budget_sheet.md Tier 2.
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
    "reporter": "remanon-correlator-13b",  # shares the 13b engine — the gpt-oss-120b
    # SHARED x2 row in D03_budget_sheet.md Tier 2; gb_saved_vs_per_agent = 10.0 GB
}


def _default_model_specs() -> dict[str, ModelSpec]:
    # weights_gb/master_gb: see docs/evidence/D03_budget_sheet.md Tier 2 per-model
    # table. gpt-oss-20b's 14.0 GB weights placeholder agrees with the Tier-1
    # measured 14.3 GiB load to within 2.1%; the other three remain COMPUTED,
    # not yet measured on real hardware.
    return {
        "remanon-triage-7b": ModelSpec("remanon-triage-7b", weights_gb=14.0, master_gb=6.0),
        "remanon-correlator-13b": ModelSpec(
            "remanon-correlator-13b", weights_gb=26.0, master_gb=10.0
        ),
        "remanon-hunter-13b": ModelSpec("remanon-hunter-13b", weights_gb=26.0, master_gb=10.0),
        "remanon-topology-7b": ModelSpec("remanon-topology-7b", weights_gb=14.0, master_gb=6.0),
    }


def _default_delta_budgets() -> dict[str, float]:
    # Per-agent working-delta budgets — a rationing design choice, not a
    # profiled measurement. Sum (26.0 GB) feeds the worst-case admission
    # check in docs/evidence/D03_budget_sheet.md Tier 2; still COMPUTED.
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

    total_capacity_gb: physical HBM3 on the MI300X — COMPUTED target, not
                        yet measured on real MI300X silicon (Tier 1's card
                        is a gfx1100, 48 GB). See docs/evidence/D03_budget_sheet.md.
    headroom_gb:       reserved slack never handed to any partition.
    models:            per-model weights + master block sizes.
    agent_delta_budget_gb: per-agent cap on working deltas; agents absent
                           from this mapping have no per-agent cap.
    """

    total_capacity_gb: float = 192.0  # MI300X HBM3 spec — COMPUTED, D03_budget_sheet.md
    headroom_gb: float = 12.0  # reserved slack — COMPUTED, D03_budget_sheet.md
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
