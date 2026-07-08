"""
Hardware & number honesty tests — Band A config + Layer L9 dashboard.

The dashboard must never present a computed/projected number as if it
were measured on real silicon. These tests pin that down mechanically:
every 192-GB-scale (MI300X-projection) figure in /api/state must carry an
explicit source=="computed" tag, the active hardware profile must be
present with its own source tag, and the six real measured-evidence
numbers must load non-zero from docs/evidence/ — never a placeholder.
"""

from __future__ import annotations

import httpx
import pytest

from app.config.evidence import load_measured_evidence
from app.config.hardware import MEASURED, PROJECTED, active_profile
from core.memory_model import DEFAULT_MODELS
from dashboard.server import DashboardSources, create_dashboard_app
from deploy.mock_engine.main import app as mock_app
from tests.test_dashboard import build_dashboard, get_state


@pytest.fixture()
def transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=mock_app)


# ---------------------------------------------------------------------------
# No 192-scale (computed/projected) number without an explicit source tag
# ---------------------------------------------------------------------------


class TestNoUnlabeledComputedFigures:
    async def test_capacity_and_headroom_are_tagged_computed(self, transport) -> None:
        _, _, dash_app = build_dashboard(transport)
        state = await get_state(dash_app)

        # capacity_gb is the MI300X 192 GB target from core/memory_model.py —
        # a real MEASURED card in this project is 48 GB, so this number is
        # never allowed to appear unlabeled.
        assert state["capacity_gb"] >= 100  # this repo's placeholder is 192.0
        assert state["capacity_source"] == "computed"
        assert state["headroom_source"] == "computed"

    async def test_every_masters_config_entry_is_tagged_computed(self, transport) -> None:
        _, _, dash_app = build_dashboard(transport)
        state = await get_state(dash_app)

        assert len(state["masters_config"]) == len(DEFAULT_MODELS)
        for entry in state["masters_config"]:
            assert entry["source"] == "computed", f"untagged memory figure: {entry}"
            assert entry["weights_gb"] > 0
            assert entry["master_gb"] > 0

    async def test_evidence_numbers_are_tagged_measured_not_computed(self, transport) -> None:
        """The six real numbers must be distinguishable from the 192-scale
        computed figures above — opposite tag, same payload."""
        _, _, dash_app = build_dashboard(transport)
        state = await get_state(dash_app)

        assert state["evidence"]["source"] == "measured"
        # the one arithmetic-derived value in the evidence block is called
        # out separately — never folded into the "measured" bucket
        assert state["evidence"]["prefix_speedup_source"] == "computed"


# ---------------------------------------------------------------------------
# Header hardware profile — source always present, never hardcodes MI300X
# ---------------------------------------------------------------------------


class TestHardwareProfilePresence:
    async def test_hardware_profile_source_present(self, transport) -> None:
        _, _, dash_app = build_dashboard(transport)
        state = await get_state(dash_app)

        profile = state["hardware_profile"]
        for key in ("name", "gfx", "vram_gb", "source", "label"):
            assert key in profile, f"missing hardware_profile key: {key}"
        assert profile["source"] in ("measured", "computed")
        assert state["hardware"] == profile["label"]

    async def test_default_profile_never_hardcodes_mi300x(self, transport) -> None:
        """Default (no REMANON_HW override) must never claim MI300X."""
        _, _, dash_app = build_dashboard(transport)
        state = await get_state(dash_app)
        assert "MI300X" not in state["hardware"]

    async def test_measured_profile_label_format(self, transport) -> None:
        rt, memory_model, _ = build_dashboard(transport)
        dash_app = create_dashboard_app(
            DashboardSources(
                event_log=rt.orchestrator.event_log,
                metrics=_metrics_for(rt, memory_model),
                memory_model=memory_model,
                residency=rt.residency,
                hardware_profile=MEASURED,
            )
        )
        state = await get_state(dash_app)
        assert state["hardware"] == "AMD gfx1100 · 48 GB · MEASURED"
        assert "MI300X" not in state["hardware"]
        assert state["hardware_profile"]["source"] == "measured"

    async def test_projected_profile_label_format(self, transport) -> None:
        rt, memory_model, _ = build_dashboard(transport)
        dash_app = create_dashboard_app(
            DashboardSources(
                event_log=rt.orchestrator.event_log,
                metrics=_metrics_for(rt, memory_model),
                memory_model=memory_model,
                residency=rt.residency,
                hardware_profile=PROJECTED,
            )
        )
        state = await get_state(dash_app)
        assert state["hardware"] == "AMD gfx942 · 192 GB · COMPUTED"
        assert state["hardware_profile"]["source"] == "computed"

    def test_active_profile_env_selection(self) -> None:
        assert active_profile({}) == MEASURED  # default, no REMANON_HW set
        assert active_profile({"REMANON_HW": "MEASURED"}) == MEASURED
        assert active_profile({"REMANON_HW": "PROJECTED"}) == PROJECTED
        assert active_profile({"REMANON_HW": "projected"}) == PROJECTED  # case-insensitive
        assert active_profile({"REMANON_HW": "garbage"}) == MEASURED  # safe fallback

    def test_measured_profile_never_claims_mi300x_scale(self) -> None:
        assert MEASURED.vram_gb == 48.0
        assert MEASURED.source == "measured"
        assert "MI300X" not in MEASURED.name
        assert PROJECTED.vram_gb == 192.0
        assert PROJECTED.source == "computed"


def _metrics_for(rt, memory_model):
    from core.budgeter import MemoryBudgeter
    from core.metrics import CoreMetrics

    return CoreMetrics(
        memory_model=memory_model,
        materializer=rt.materializer,
        residency=rt.residency,
        budgeter=MemoryBudgeter(memory_model),
    )


# ---------------------------------------------------------------------------
# Measured evidence — must load non-zero real numbers from docs/evidence/
# ---------------------------------------------------------------------------


class TestMeasuredEvidenceLoads:
    def test_loads_nonzero_from_real_evidence_files(self) -> None:
        evidence = load_measured_evidence()

        assert evidence.available
        assert evidence.weights_gib is not None and evidence.weights_gib > 0
        assert evidence.kv_cache_gib is not None and evidence.kv_cache_gib > 0
        assert evidence.kv_cache_tokens is not None and evidence.kv_cache_tokens > 0
        assert evidence.prefix_hit_rate_pct is not None and evidence.prefix_hit_rate_pct > 0
        assert evidence.cold_s is not None and evidence.cold_s > 0
        assert evidence.warm_s is not None and evidence.warm_s > 0
        assert evidence.warm2_s is not None and evidence.warm2_s > 0
        assert evidence.vram_used_gb is not None and evidence.vram_used_gb > 0

    def test_prefix_speedup_is_computed_from_measured_cold_warm(self) -> None:
        evidence = load_measured_evidence()
        assert evidence.prefix_speedup is not None
        assert evidence.prefix_speedup == round(evidence.cold_s / evidence.warm_s, 1)

    def test_missing_evidence_dir_returns_none_never_a_placeholder(self, tmp_path) -> None:
        evidence = load_measured_evidence(tmp_path)
        assert not evidence.available
        assert evidence.weights_gib is None
        assert evidence.kv_cache_gib is None
        assert evidence.vram_used_gb is None

    async def test_state_evidence_matches_loader(self, transport) -> None:
        _, _, dash_app = build_dashboard(transport)
        state = await get_state(dash_app)
        evidence = load_measured_evidence()

        assert state["evidence"]["weights_gib"] == evidence.weights_gib
        assert state["evidence"]["kv_cache_gib"] == evidence.kv_cache_gib
        assert state["evidence"]["kv_cache_tokens"] == evidence.kv_cache_tokens
        assert state["evidence"]["vram_used_gb"] == evidence.vram_used_gb
