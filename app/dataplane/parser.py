"""
HDFS log parser — Band C, Layer L5.

Converts raw HDFS log lines into ParsedLine dataclasses.
Malformed lines are counted and skipped; the parser never raises.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# HDFS line format: <YYMMDD> <HHMMSS> <pid> <LEVEL> <Component>: <message>
_LINE_RE = re.compile(r"^(\d{6})\s+(\d{6})\s+(\d+)\s+([A-Z]+)\s+([^\s][^:]*?):\s+(.+)$")


@dataclass(frozen=True, slots=True)
class ParsedLine:
    date_str: str  # raw YYMMDD
    time_str: str  # raw HHMMSS
    pid: int
    level: str
    component: str
    message: str
    raw_line: str


@dataclass
class ParseResult:
    records: list[ParsedLine]
    skipped: int


def parse_lines(lines: Iterator[str]) -> ParseResult:
    """Parse an iterable of raw log lines, skipping malformed ones."""
    records: list[ParsedLine] = []
    skipped = 0
    for raw in lines:
        line = raw.rstrip("\n\r")
        if not line.strip():
            continue
        m = _LINE_RE.match(line)
        if m is None:
            skipped += 1
            continue
        date_str, time_str, pid_str, level, component, message = m.groups()
        records.append(
            ParsedLine(
                date_str=date_str,
                time_str=time_str,
                pid=int(pid_str),
                level=level.upper(),
                component=component,
                message=message,
                raw_line=line,
            )
        )
    return ParseResult(records=records, skipped=skipped)


def parse_file(path: Path) -> ParseResult:
    """Convenience wrapper: open a log file and parse all lines."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        return parse_lines(fh)
