"""Band B — residency: lease-pinned block management (Memory Arbiter)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    """Will track HBM3-vs-host-RAM tensor regions once engine-level enforcement
    lands (the research roadmap); intentionally deferred — no runtime path
    depends on it today."""

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


# ---------------------------------------------------------------------------
# Lease-pinned residency (Phase 3 — Memory Arbiter)
# ---------------------------------------------------------------------------


class PinnedBlockError(RuntimeError):
    """Raised when evict() targets a block that holds >= 1 active lease."""


@dataclass(frozen=True, slots=True)
class Lease:
    lease_id: str
    agent: str
    model: str
    context_id: str
    created_at: datetime


@dataclass(slots=True)
class Block:
    model: str
    context_id: str


class LeaseTable:
    """Index of active leases, keyed by lease_id, with per-model lookups."""

    def __init__(self) -> None:
        self._by_id: dict[str, Lease] = {}

    def add(self, lease: Lease) -> None:
        self._by_id[lease.lease_id] = lease

    def remove(self, lease_id: str) -> Lease:
        if lease_id not in self._by_id:
            raise KeyError(f"Unknown lease: {lease_id}")
        return self._by_id.pop(lease_id)

    def get(self, lease_id: str) -> Lease:
        if lease_id not in self._by_id:
            raise KeyError(f"Unknown lease: {lease_id}")
        return self._by_id[lease_id]

    def active(self) -> list[Lease]:
        return list(self._by_id.values())

    def leases_for_model(self, model: str) -> list[Lease]:
        return [le for le in self._by_id.values() if le.model == model]

    def agents_for_model(self, model: str) -> set[str]:
        return {le.agent for le in self._by_id.values() if le.model == model}

    def __len__(self) -> int:
        return len(self._by_id)


class ResidencyManager:
    """
    Manages master-context blocks and the leases pinning them.

    INVARIANT: a block with >= 1 active lease is PINNED. evict() is the only
    eviction code path, and it raises PinnedBlockError for pinned blocks —
    nothing may bypass this.

    The optional on_release callback fires after a lease is released
    (used to free the lease's delta budget in the MemoryBudgeter).
    """

    def __init__(self, on_release: Callable[[Lease], None] | None = None) -> None:
        self._lock = asyncio.Lock()
        self._blocks: dict[str, Block] = {}
        self._table = LeaseTable()
        self._on_release = on_release
        # Cumulative sharing history: once an agent has leased a model's
        # master it has ridden the shared copy — releasing the lease does not
        # undo that fact. Append-only; feeds gb_saved_vs_per_agent.
        self._ever_leased: dict[str, set[str]] = {}

    async def lease(self, context_id: str, model: str, agent: str) -> Lease:
        async with self._lock:
            self._blocks.setdefault(model, Block(model=model, context_id=context_id))
            self._ever_leased.setdefault(model, set()).add(agent)
            new_lease = Lease(
                lease_id=uuid.uuid4().hex,
                agent=agent,
                model=model,
                context_id=context_id,
                created_at=datetime.now(UTC),
            )
            self._table.add(new_lease)
            return new_lease

    async def release(self, lease_id: str) -> None:
        async with self._lock:
            released = self._table.remove(lease_id)
        if self._on_release is not None:
            self._on_release(released)

    async def evict(self, model: str) -> None:
        """Remove an unpinned block. Raises PinnedBlockError if any lease is active."""
        async with self._lock:
            if model not in self._blocks:
                raise KeyError(f"No block for model: {model!r}")
            holders = self._table.leases_for_model(model)
            if holders:
                raise PinnedBlockError(
                    f"Block {model!r} is pinned by {len(holders)} active lease(s); eviction refused"
                )
            del self._blocks[model]

    # --- read-only queries (no lock needed: asyncio single-threaded, no awaits) ---

    def is_pinned(self, model: str) -> bool:
        return bool(self._table.leases_for_model(model))

    def has_block(self, model: str) -> bool:
        return model in self._blocks

    def agents_sharing(self, model: str) -> set[str]:
        """Agents holding an ACTIVE lease on *model* right now (live view)."""
        return self._table.agents_for_model(model)

    def agents_ever_leased(self, model: str) -> set[str]:
        """
        Every distinct agent that has ever leased *model*'s master (cumulative).

        This is the correct input for gb_saved_vs_per_agent: the saving from a
        shared ride happened when the lease was taken and is not reversed by
        releasing it.
        """
        return set(self._ever_leased.get(model, set()))

    @property
    def active_leases(self) -> int:
        return len(self._table)

    @property
    def lease_table(self) -> LeaseTable:
        return self._table
