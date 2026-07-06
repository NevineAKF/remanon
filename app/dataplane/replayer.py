"""
Telemetry replayer — Band C, Layer L5.

Async generator that streams TelemetryRecord objects in original chronological
order with configurable time compression.  The callback parameter is the
future L8 Intake hook; pass None to get a plain async iterator.

Time compression:
    speed_factor = N  →  N log-seconds elapse per 1 wall-second
    Default 60.0  →  1 real hour of log time = 1 demo minute of wall time
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from app.dataplane.normalizer import TelemetryRecord


@dataclass
class ReplayStats:
    emitted: int = 0
    total_wall_sleep_s: float = 0.0


async def stream(
    records: list[TelemetryRecord],
    *,
    speed_factor: float = 60.0,
    callback: Callable[[TelemetryRecord], Awaitable[None]] | None = None,
    max_sleep_s: float = 5.0,
) -> AsyncIterator[TelemetryRecord]:
    """
    Yield *records* in chronological order, sleeping proportionally between
    events to imitate original timing compressed by *speed_factor*.

    Parameters
    ----------
    records:
        Source records (need not be pre-sorted).
    speed_factor:
        How many log-seconds map to one wall-second.
        60.0 → 1 real hour becomes 1 demo minute.
    callback:
        Async callable invoked with each record *before* it is yielded.
        Intended for the future L8 Intake pipeline.
    max_sleep_s:
        Cap on any single sleep to prevent the generator from stalling on
        large log gaps.
    """
    if not records:
        return

    sorted_recs = sorted(records, key=lambda r: r.ts)
    prev_ts = None

    for rec in sorted_recs:
        if prev_ts is not None:
            log_delta_s = (rec.ts - prev_ts).total_seconds()
            wall_sleep = min(log_delta_s / speed_factor, max_sleep_s)
            if wall_sleep > 0:
                await asyncio.sleep(wall_sleep)
        prev_ts = rec.ts

        if callback is not None:
            await callback(rec)

        yield rec


def compute_wall_duration(
    records: list[TelemetryRecord],
    speed_factor: float = 60.0,
    max_sleep_s: float = 5.0,
) -> float:
    """
    Return the total wall-clock sleep in seconds that stream() would
    accumulate for *records* — useful for testing without actually awaiting.
    """
    if len(records) < 2:
        return 0.0
    sorted_recs = sorted(records, key=lambda r: r.ts)
    total = 0.0
    for a, b in zip(sorted_recs, sorted_recs[1:], strict=False):
        log_delta = (b.ts - a.ts).total_seconds()
        total += min(log_delta / speed_factor, max_sleep_s)
    return total
