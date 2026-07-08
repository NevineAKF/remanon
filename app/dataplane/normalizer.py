"""
Telemetry normalizer — Band A, Layer L5.

Converts dialect-specific ParsedLine objects into canonical TelemetryRecord
instances.  New dialects (Thunderbird, BGL, …) plug in by subclassing
BaseNormalizer without touching any existing code.

HDFS goes through the ParsedLine pipeline (parser.py's HDFS-shaped regex).
The other Loghub dialects below have genuinely different raw formats (BGL's
label+epoch+date+node+time+node-repeat+type+component+level+content; Spark/
Hadoop's date+time+level[+thread]+component+message; Thunderbird's syslog
free text with no explicit level field) that don't fit ParsedLine's
HDFS-shaped schema without inventing fields that aren't in the real data.
Forcing them through it would mean fabricating values (e.g. a fake pid) —
exactly what this module must never do. Instead each of these subclasses
implements parse_raw(raw_line) directly: real regex parsing of the real
line, returning None (skip, counted, never fabricated) on no match.
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
    dialect: str  # "hdfs" | "bgl" | "thunderbird" | "spark" | "hadoop" — also the
    # system-of-origin column: one dialect per real Loghub system, so
    # `dialect` IS the "system" distinguisher a combined multi-system
    # store needs (see app/dataplane/ingest.py's `--dataset all`).
    raw_line: str  # original untouched line


# ---------------------------------------------------------------------------
# Normalizer contract
# ---------------------------------------------------------------------------


class BaseNormalizer(ABC):
    dialect: str = ""

    @abstractmethod
    def normalize(self, parsed: ParsedLine) -> TelemetryRecord:
        """Convert a ParsedLine into a canonical TelemetryRecord (HDFS-shaped dialects)."""

    def parse_raw(self, raw_line: str) -> TelemetryRecord | None:
        """
        Parse + normalize one raw log line directly, for dialects whose real
        format doesn't fit ParsedLine's HDFS-shaped schema. Returns None for
        a line that doesn't match this dialect's real format — skipped and
        counted by the caller, never filled with invented values.

        Base implementation raises: only override this on dialects that use
        this path instead of the ParsedLine pipeline.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement parse_raw()")


# ---------------------------------------------------------------------------
# HDFS dialect — unchanged
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
# BGL dialect — real format (space-separated, 9 fixed fields + free content):
#   Label Epoch Date Node Time NodeRepeat Type Component Level Content
# e.g. "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872
#       R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
# Label != "-" marks a real Loghub-annotated anomaly line (e.g. APPREAD, KERNDTLB).
# ---------------------------------------------------------------------------


class BGLNormalizer(BaseNormalizer):
    dialect = "bgl"

    _LINE_RE = re.compile(
        r"^(?P<label>\S+)\s+(?P<epoch>\d+)\s+(?P<date>\S+)\s+(?P<node>\S+)\s+"
        r"(?P<time>\S+)\s+(?P<node2>\S+)\s+(?P<rtype>\S+)\s+(?P<component>\S+)\s+"
        r"(?P<level>\S+)\s+(?P<content>.*)$"
    )
    _LEVEL_MAP = {
        "INFO": "INFO",
        "WARNING": "WARN",
        "ERROR": "ERROR",
        "FATAL": "FATAL",
        "SEVERE": "ERROR",
    }

    def normalize(self, parsed: ParsedLine) -> TelemetryRecord:
        raise NotImplementedError("BGLNormalizer uses parse_raw(), not the ParsedLine pipeline")

    def parse_raw(self, raw_line: str) -> TelemetryRecord | None:
        line = raw_line.rstrip("\n\r")
        m = self._LINE_RE.match(line)
        if m is None:
            return None
        try:
            ts = datetime.fromtimestamp(int(m["epoch"]), tz=UTC)
        except (ValueError, OSError):
            return None
        level = self._LEVEL_MAP.get(m["level"].upper(), "INFO")
        return TelemetryRecord(
            ts=ts,
            node=m["node"],
            level=level,
            component=f"{m['rtype']}.{m['component']}",
            message=m["content"].strip(),
            dialect=self.dialect,
            raw_line=line,
        )


# ---------------------------------------------------------------------------
# Thunderbird dialect — real format (syslog-style, no explicit level field):
#   Label Epoch Date Node Month Day Time Node2 "component[pid]: message"
# e.g. "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228
#       crond(pam_unix)[2915]: session closed for user root"
# The Loghub 2k sample carries no non-"-" labeled (anomaly) lines, so a real
# severity signal has to come from the message text itself — the same
# keyword-classification approach standard log tooling (e.g. syslog
# ingesters) uses when a source emits no structured severity. This reads
# real words in the real message; it never invents content.
# ---------------------------------------------------------------------------


class ThunderbirdNormalizer(BaseNormalizer):
    dialect = "thunderbird"

    _LINE_RE = re.compile(
        r"^(?P<label>\S+)\s+(?P<epoch>\d+)\s+(?P<date>\S+)\s+(?P<node>\S+)\s+"
        r"(?P<month>\S+)\s+(?P<day>\S+)\s+(?P<time>\S+)\s+(?P<node2>\S+)\s+(?P<content>.*)$"
    )
    _FATAL_RE = re.compile(r"\bfatal\b", re.IGNORECASE)
    _ERROR_RE = re.compile(r"\b(error|fail(?:ed|ure)?)\b", re.IGNORECASE)
    _WARN_RE = re.compile(r"\b(warn(?:ing)?|denied|invalid|refused)\b", re.IGNORECASE)

    def normalize(self, parsed: ParsedLine) -> TelemetryRecord:
        raise NotImplementedError(
            "ThunderbirdNormalizer uses parse_raw(), not the ParsedLine pipeline"
        )

    def parse_raw(self, raw_line: str) -> TelemetryRecord | None:
        line = raw_line.rstrip("\n\r")
        m = self._LINE_RE.match(line)
        if m is None:
            return None
        try:
            ts = datetime.fromtimestamp(int(m["epoch"]), tz=UTC)
        except (ValueError, OSError):
            return None
        content = m["content"].strip()
        component, _, message = content.partition(": ")
        if not message:
            component, message = "syslog", content
        if m["label"] != "-":
            level = "WARN"  # Loghub flagged this line; exact severity tier unstated
        elif self._FATAL_RE.search(content):
            level = "FATAL"
        elif self._ERROR_RE.search(content):
            level = "ERROR"
        elif self._WARN_RE.search(content):
            level = "WARN"
        else:
            level = "INFO"
        return TelemetryRecord(
            ts=ts,
            node=m["node"],
            level=level,
            component=component,
            message=message,
            dialect=self.dialect,
            raw_line=line,
        )


# ---------------------------------------------------------------------------
# Spark dialect — real format:
#   YY/MM/DD HH:MM:SS LEVEL Component: Message
# e.g. "17/06/09 20:10:40 INFO executor.CoarseGrainedExecutorBackend:
#       Registered signal handlers for [TERM, HUP, INT]"
# ---------------------------------------------------------------------------


class SparkNormalizer(BaseNormalizer):
    dialect = "spark"

    _LINE_RE = re.compile(
        r"^(?P<date>\d{2}/\d{2}/\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
        r"(?P<level>[A-Z]+)\s+(?P<component>[^\s][^:]*?):\s+(?P<message>.+)$"
    )
    _LEVEL_MAP = {"INFO": "INFO", "WARN": "WARN", "ERROR": "ERROR", "FATAL": "FATAL"}

    def normalize(self, parsed: ParsedLine) -> TelemetryRecord:
        raise NotImplementedError("SparkNormalizer uses parse_raw(), not the ParsedLine pipeline")

    def parse_raw(self, raw_line: str) -> TelemetryRecord | None:
        line = raw_line.rstrip("\n\r")
        m = self._LINE_RE.match(line)
        if m is None:
            return None
        try:
            ts = datetime.strptime(f"{m['date']} {m['time']}", "%y/%m/%d %H:%M:%S").replace(
                tzinfo=UTC
            )
        except ValueError:
            return None
        message = m["message"].strip()
        node = _extract_ip(message) or _extract_ip(m["component"]) or "unknown"
        return TelemetryRecord(
            ts=ts,
            node=node,
            level=self._LEVEL_MAP.get(m["level"].upper(), "INFO"),
            component=m["component"],
            message=message,
            dialect=self.dialect,
            raw_line=line,
        )


# ---------------------------------------------------------------------------
# Hadoop dialect — real format:
#   YYYY-MM-DD HH:MM:SS,mmm LEVEL [Thread] Component: Message
# e.g. "2015-10-18 18:01:47,978 INFO [main]
#       org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster ..."
# ---------------------------------------------------------------------------


class HadoopNormalizer(BaseNormalizer):
    dialect = "hadoop"

    _LINE_RE = re.compile(
        r"^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2},\d{3})\s+"
        r"(?P<level>[A-Z]+)\s+\[(?P<thread>[^\]]*)\]\s+(?P<component>[^:]+):\s+(?P<message>.*)$"
    )
    _LEVEL_MAP = {"INFO": "INFO", "WARN": "WARN", "ERROR": "ERROR", "FATAL": "FATAL"}

    def normalize(self, parsed: ParsedLine) -> TelemetryRecord:
        raise NotImplementedError("HadoopNormalizer uses parse_raw(), not the ParsedLine pipeline")

    def parse_raw(self, raw_line: str) -> TelemetryRecord | None:
        line = raw_line.rstrip("\n\r")
        m = self._LINE_RE.match(line)
        if m is None:
            return None
        try:
            ts = datetime.strptime(f"{m['date']} {m['time']}", "%Y-%m-%d %H:%M:%S,%f").replace(
                tzinfo=UTC
            )
        except ValueError:
            return None
        message = m["message"].strip()
        node = _extract_ip(message) or _extract_ip(m["component"]) or "unknown"
        return TelemetryRecord(
            ts=ts,
            node=node,
            level=self._LEVEL_MAP.get(m["level"].upper(), "INFO"),
            component=m["component"].strip(),
            message=message,
            dialect=self.dialect,
            raw_line=line,
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
    "bgl": BGLNormalizer(),
    "thunderbird": ThunderbirdNormalizer(),
    "spark": SparkNormalizer(),
    "hadoop": HadoopNormalizer(),
}


def get_normalizer(dialect: str) -> BaseNormalizer:
    if dialect not in NORMALIZERS:
        raise KeyError(f"Unknown dialect: {dialect!r}. Available: {list(NORMALIZERS)}")
    return NORMALIZERS[dialect]
