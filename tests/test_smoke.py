"""Smoke tests — must pass with zero GPU and no external services."""

from __future__ import annotations

import uuid
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from contracts.contract_a import Artifact, _load_schema
from core.budgeter import Budget, Budgeter
from core.registry import AgentHandle, Registry
from core.residency import RegionRecord, ResidencyState, ResidencyTracker
from deploy.mock_engine.main import app as mock_app

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_triage_raw(routing: list[str] | None = None) -> dict:
    return {
        "artifact_id": str(uuid.uuid4()),
        "agent": "triage",
        "version": "0.1.0",
        "payload": {
            "severity": "high",
            "category": "lateral_movement",
            "summary": "Suspicious east-west traffic detected.",
            "routing": routing or ["correlator", "hunter"],
            "confidence": 0.91,
            "hbm3_handle": None,
        },
    }


# ---------------------------------------------------------------------------
# Contract A — schema validation
# ---------------------------------------------------------------------------


class TestArtifactSchemas:
    def test_all_schemas_present(self):
        schema_dir = Path("contracts/artifact_schemas")
        expected = {"triage", "correlator", "hunter", "topology", "reporter"}
        found = {p.stem for p in schema_dir.glob("*.json")}
        assert expected == found

    def test_triage_schema_is_valid_json(self):
        schema = _load_schema("triage")
        assert schema["$id"] == "remanon/artifact/triage/v1"

    def test_valid_triage_artifact_passes_validation(self):
        raw = _make_triage_raw()
        artifact = Artifact(raw, "triage")
        assert artifact.agent == "triage"
        assert artifact.hbm3_handle is None

    def test_invalid_severity_raises(self):
        raw = _make_triage_raw()
        raw["payload"]["severity"] = "EXPLODING"
        with pytest.raises(jsonschema.ValidationError):
            Artifact(raw, "triage")

    def test_missing_routing_raises(self):
        raw = _make_triage_raw()
        del raw["payload"]["routing"]
        with pytest.raises(jsonschema.ValidationError):
            Artifact(raw, "triage")


# ---------------------------------------------------------------------------
# Core — Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_retrieve_artifact(self):
        reg = Registry()
        artifact = Artifact(_make_triage_raw(), "triage")
        reg.register_artifact(artifact)
        retrieved = reg.get_artifact(artifact.artifact_id)
        assert retrieved.artifact_id == artifact.artifact_id

    def test_list_artifacts_by_agent(self):
        reg = Registry()
        reg.register_artifact(Artifact(_make_triage_raw(), "triage"))
        reg.register_artifact(Artifact(_make_triage_raw(), "triage"))
        assert len(reg.list_artifacts(agent="triage")) == 2
        assert len(reg.list_artifacts(agent="hunter")) == 0

    def test_register_and_retrieve_agent_handle(self):
        reg = Registry()
        handle = AgentHandle(name="triage", model_uri="hf://remanon/triage-7b")
        reg.register_agent(handle)
        assert reg.get_agent("triage").model_uri == "hf://remanon/triage-7b"

    def test_missing_artifact_raises_key_error(self):
        reg = Registry()
        with pytest.raises(KeyError):
            reg.get_artifact("does-not-exist")


# ---------------------------------------------------------------------------
# Core — ResidencyTracker
# ---------------------------------------------------------------------------


class TestResidencyTracker:
    def test_track_and_query(self):
        tracker = ResidencyTracker()
        rec = RegionRecord(
            handle="h-001",
            name="triage_weights",
            size_bytes=8 * 1024**3,
            state=ResidencyState.HBM3,
        )
        tracker.track(rec)
        assert tracker.get("h-001").state == ResidencyState.HBM3

    def test_total_hbm3_bytes(self):
        tracker = ResidencyTracker()
        tracker.track(RegionRecord("h1", "a", 4 * 1024**3, ResidencyState.HBM3))
        tracker.track(RegionRecord("h2", "b", 2 * 1024**3, ResidencyState.HOST_RAM))
        assert tracker.total_hbm3_bytes() == 4 * 1024**3

    def test_update_state(self):
        tracker = ResidencyTracker()
        tracker.track(RegionRecord("h1", "a", 1, ResidencyState.HBM3))
        tracker.update_state("h1", ResidencyState.EVICTED)
        assert tracker.get("h1").state == ResidencyState.EVICTED


# ---------------------------------------------------------------------------
# Core — Budgeter
# ---------------------------------------------------------------------------


class TestBudgeter:
    def test_budget_check_within_limit(self):
        b = Budgeter()
        b.set_budget(Budget(agent="triage", max_tokens=1024))
        b.record_usage("triage", tokens=500)
        assert b.check_token_budget("triage", 400) is True

    def test_budget_check_exceeds_limit(self):
        b = Budgeter()
        b.set_budget(Budget(agent="triage", max_tokens=1024))
        b.record_usage("triage", tokens=900)
        assert b.check_token_budget("triage", 200) is False

    def test_summary_fields(self):
        b = Budgeter()
        b.set_budget(Budget(agent="hunter", max_tokens=2048))
        b.record_usage("hunter", tokens=128, hbm3_bytes=1024)
        summary = b.summary()
        assert len(summary) == 1
        assert summary[0]["agent"] == "hunter"
        assert summary[0]["tokens_used"] == 128


# ---------------------------------------------------------------------------
# Mock engine — Contract B HTTP surface
# ---------------------------------------------------------------------------


class TestMockEngine:
    @pytest.fixture(autouse=True)
    def client(self):
        self.client = TestClient(mock_app)

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["gpu"] is False

    def test_list_models(self):
        resp = self.client.get("/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["data"]]
        assert "remanon-triage-7b" in ids
        assert len(ids) == 5

    def test_chat_completion_returns_mock_text(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "remanon-triage-7b",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "remanon-triage-7b"
        assert "[MOCK ENGINE]" in body["choices"][0]["message"]["content"]
        assert body["usage"]["total_tokens"] > 0

    def test_unknown_model_returns_404(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404

    def test_streaming_not_supported(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "remanon-triage-7b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 400
