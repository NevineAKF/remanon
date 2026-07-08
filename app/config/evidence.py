"""
Measured-evidence loader — Band A config layer.

Parses the real, human-authored evidence files in docs/evidence/ into
numbers the dashboard can show. This module NEVER invents a value: a
missing file or an unparseable field comes back as None, and callers must
render that as unavailable — not silently substitute a placeholder. The
one computed exception (prefix_speedup) is derived by simple arithmetic
from two measured values in the same file and is labeled as computed,
never folded into the measured set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_EVIDENCE_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "evidence"

_KV_HITRATE_FILE = "kv_and_hitrate.txt"
_PREFIX_REUSE_FILE = "prefix_reuse.txt"
_VRAM_AFTER_FILE = "vram_after.txt"

_KV_RE = re.compile(r"^([a-z0-9_]+)\s*:\s*(.+)$")
_VRAM_BYTES_RE = re.compile(r"VRAM Total Used Memory \(B\):\s*(\d+)")


def _parse_kv_file(path: Path) -> dict[str, float]:
    """Parse simple `key: number` lines; ignores narrative/header lines."""
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _KV_RE.match(line.strip())
        if not m:
            continue
        key, raw_val = m.group(1), m.group(2).strip()
        try:
            out[key] = float(raw_val)
        except ValueError:
            continue
    return out


@dataclass(frozen=True, slots=True)
class MeasuredEvidence:
    """
    All fields None-safe: a viewer must never mistake "the file was
    missing" for "the measurement was zero". `available` is False only
    when every source file was missing (nothing to show at all).
    """

    weights_gib: float | None = None
    kv_cache_gib: float | None = None
    kv_cache_tokens: int | None = None
    prefix_hit_rate_pct: float | None = None
    cold_s: float | None = None
    warm_s: float | None = None
    warm2_s: float | None = None
    prefix_speedup: float | None = None  # COMPUTED = cold_s / warm_s
    vram_used_gb: float | None = None
    source_files: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return bool(self.source_files)

    def to_dict(self) -> dict[str, object]:
        return {
            "weights_gib": self.weights_gib,
            "kv_cache_gib": self.kv_cache_gib,
            "kv_cache_tokens": self.kv_cache_tokens,
            "prefix_hit_rate_pct": self.prefix_hit_rate_pct,
            "cold_s": self.cold_s,
            "warm_s": self.warm_s,
            "warm2_s": self.warm2_s,
            "prefix_speedup": self.prefix_speedup,
            "vram_used_gb": self.vram_used_gb,
            "source": "measured",
            "prefix_speedup_source": "computed",  # the one derived field, called out
            "source_files": list(self.source_files),
        }


def load_measured_evidence(evidence_dir: Path = _EVIDENCE_DIR) -> MeasuredEvidence:
    kv_path = evidence_dir / _KV_HITRATE_FILE
    prefix_path = evidence_dir / _PREFIX_REUSE_FILE
    vram_path = evidence_dir / _VRAM_AFTER_FILE

    kv = _parse_kv_file(kv_path)
    prefix = _parse_kv_file(prefix_path)

    vram_used_gb: float | None = None
    sources: list[str] = []
    if kv_path.exists():
        sources.append(_KV_HITRATE_FILE)
    if prefix_path.exists():
        sources.append(_PREFIX_REUSE_FILE)
    if vram_path.exists():
        sources.append(_VRAM_AFTER_FILE)
        m = _VRAM_BYTES_RE.search(vram_path.read_text(encoding="utf-8"))
        if m:
            vram_used_gb = round(int(m.group(1)) / 1_000_000_000, 2)

    cold_s = prefix.get("cold_s")
    warm_s = prefix.get("warm_s")
    prefix_speedup = round(cold_s / warm_s, 1) if cold_s and warm_s else None

    kv_tokens = kv.get("kv_cache_tokens")

    return MeasuredEvidence(
        weights_gib=kv.get("weights_gib"),
        kv_cache_gib=kv.get("kv_cache_gib"),
        kv_cache_tokens=int(kv_tokens) if kv_tokens is not None else None,
        prefix_hit_rate_pct=kv.get("prefix_hit_rate_pct"),
        cold_s=cold_s,
        warm_s=warm_s,
        warm2_s=prefix.get("warm2_s"),
        prefix_speedup=prefix_speedup,
        vram_used_gb=vram_used_gb,
        source_files=tuple(sources),
    )
