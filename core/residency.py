"""Band B — GPU-memory residency tracking (stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ResidencyState(Enum):
    HBM3 = auto()
    HOST_RAM = auto()
    EVICTED = auto()
    UNKNOWN = auto()


@dataclass
class RegionRecord:
    handle: str
    name: str
    size_bytes: int
    state: ResidencyState = ResidencyState.UNKNOWN
    metadata: dict = field(default_factory=dict)


class ResidencyTracker:
    """Tracks which tensor regions reside in HBM3 vs host RAM (stub)."""

    def __init__(self) -> None:
        self._regions: dict[str, RegionRecord] = {}

    def track(self, record: RegionRecord) -> None:
        self._regions[record.handle] = record

    def update_state(self, handle: str, state: ResidencyState) -> None:
        if handle not in self._regions:
            raise KeyError(f"Unknown handle: {handle}")
        self._regions[handle].state = state

    def get(self, handle: str) -> RegionRecord:
        if handle not in self._regions:
            raise KeyError(f"Unknown handle: {handle}")
        return self._regions[handle]

    def list_by_state(self, state: ResidencyState) -> list[RegionRecord]:
        return [r for r in self._regions.values() if r.state == state]

    def total_hbm3_bytes(self) -> int:
        return sum(r.size_bytes for r in self._regions.values() if r.state == ResidencyState.HBM3)
