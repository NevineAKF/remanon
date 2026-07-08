"""
Hardware profile — Band A config layer.

The dashboard must never present one number as if it came from real
silicon when it didn't. This module is the single source of truth for
"what hardware is this actually running on right now": MEASURED describes
the real AMD card this project has run vLLM on (see docs/evidence/);
PROJECTED describes the MI300X capacity core/memory_model.py is sized for
but has not yet run on. The active profile is chosen by an env var, never
hardcoded into the dashboard — see docs/evidence/D03_budget_sheet.md for
the full measured-vs-computed accounting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_ENV_VAR = "REMANON_HW"


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    name: str
    gfx: str
    vram_gb: float
    source: str  # "measured" | "computed"

    @property
    def label(self) -> str:
        """e.g. 'AMD gfx1100 · 48 GB · MEASURED' — the header's honest hardware tag."""
        return f"AMD {self.gfx} · {self.vram_gb:g} GB · {self.source.upper()}"


MEASURED = HardwareProfile(
    name="AMD Radeon PRO W7900-class", gfx="gfx1100", vram_gb=48.0, source="measured"
)
PROJECTED = HardwareProfile(
    name="AMD Instinct MI300X", gfx="gfx942", vram_gb=192.0, source="computed"
)

_PROFILES: dict[str, HardwareProfile] = {"MEASURED": MEASURED, "PROJECTED": PROJECTED}


def active_profile(env: dict[str, str] | None = None) -> HardwareProfile:
    """
    The active hardware profile, selected via the REMANON_HW env var
    (MEASURED | PROJECTED, default MEASURED — never defaults to claiming
    the unmeasured MI300X). Pass `env` explicitly in tests instead of
    monkeypatching os.environ.
    """
    source = env if env is not None else os.environ
    key = source.get(_ENV_VAR, "MEASURED").strip().upper()
    return _PROFILES.get(key, MEASURED)
