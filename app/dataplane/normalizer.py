"""
Telemetry normalizer — Band A, Layer L5.

Converts dialect-specific ParsedLine objects into canonical TelemetryRecord
instances.  New dialects (Thunderbird, BGL, …) plug in by subclassing
BaseNormalizer without touching any existing code.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

from app.dataplane.parser import ParsedLine

# Match the first bare IP address in a log message, e.g. 10.250.19.102
_IP_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")


# ---------------------------------------------------------------------------
# Canonical record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    ts: datetime  # UTC-aware
    node: str  # source IP or hostname
    level: str  # INFO / WARN / ERROR / FATAL
    component: str  # qualified class name
    message: str  # trimmed log message
    dialect: str  # "hdfs" | "thunderbird" | "bgl" | …
    raw_line: str  # original untouched line


# ---------------------------------------------------------------------------
# Normalizer contract
# ---------------------------------------------------------------------------


class BaseNormalizer(ABC):
    dialect: str = ""

    @abstractmethod
    def normalize(self, parsed: ParsedLine) -> TelemetryRecord:
        """Convert a ParsedLine into a canonical TelemetryRecord."""


# ---------------------------------------------------------------------------
# HDFS dialect
# ---------------------------------------------------------------------------


class HDFSNormalizer(BaseNormalizer):
    dialect = "hdfs"

    def normalize(self, parsed: ParsedLine) -> TelemetryRecord:
        ts = _parse_hdfs_ts(parsed.date_str, parsed.time_str)
        node = _extract_ip(parsed.message) or _extract_ip(parsed.component) or "unknown"
        return TelemetryRecord(
            ts=ts,
            node=node,
            level=parsed.level,
            component=parsed.component,
            message=parsed.message,
            dialect=self.dialect,
            raw_line=parsed.raw_line,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_hdfs_ts(date_str: str, time_str: str) -> datetime:
    """Parse YYMMDD + HHMMSS into a UTC-aware datetime (year treated as 20YY)."""
    yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
    hh, mi, ss = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
    return datetime(2000 + yy, mm, dd, hh, mi, ss, tzinfo=UTC)


def _extract_ip(text: str) -> str | None:
    m = _IP_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Registry of known normalizers (extend here when adding dialects)
# ---------------------------------------------------------------------------

NORMALIZERS: dict[str, BaseNormalizer] = {
    "hdfs": HDFSNormalizer(),
}


def get_normalizer(dialect: str) -> BaseNormalizer:
    if dialect not in NORMALIZERS:
        raise KeyError(f"Unknown dialect: {dialect!r}. Available: {list(NORMALIZERS)}")
    return NORMALIZERS[dialect]
