"""Band B — artifact catalog + engine registry (Memory Arbiter)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from app.adapter.contract_b_client import ContractBClient
from contracts.contract_a import AgentName, Artifact, ArtifactId
from core.memory_model import DEFAULT_MODELS


@dataclass
class AgentHandle:
    name: AgentName
    hbm3_handle: str | None = None
    model_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry:
    """In-process registry — to be backed by shared HBM3 in production."""

    def __init__(self) -> None:
        self._artifacts: dict[ArtifactId, Artifact] = {}
        self._agents: dict[AgentName, AgentHandle] = {}

    # --- artifact operations ---

    def register_artifact(self, artifact: Artifact) -> None:
        self._artifacts[artifact.artifact_id] = artifact

    def get_artifact(self, artifact_id: ArtifactId) -> Artifact:
        if artifact_id not in self._artifacts:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return self._artifacts[artifact_id]

    def list_artifacts(self, agent: AgentName | None = None) -> list[Artifact]:
        arts = list(self._artifacts.values())
        if agent:
            arts = [a for a in arts if a.agent == agent]
        return arts

    # --- agent handle operations ---

    def register_agent(self, handle: AgentHandle) -> None:
        self._agents[handle.name] = handle

    def get_agent(self, name: AgentName) -> AgentHandle:
        if name not in self._agents:
            raise KeyError(f"Agent not registered: {name}")
        return self._agents[name]

    def list_agents(self) -> list[AgentHandle]:
        return list(self._agents.values())


# ---------------------------------------------------------------------------
# Engine registry (Phase 3 — Memory Arbiter)
# ---------------------------------------------------------------------------


@dataclass
class Engine:
    """One inference engine serving exactly one model via Contract B."""

    model: str
    base_url: str
    port: int
    healthy: bool | None = None  # None = never checked
    last_checked: datetime | None = None


class EngineRegistry:
    """
    Maps model name → engine endpoint and tracks health via Contract B.

    An optional httpx transport is threaded through to the Contract B client
    so tests can target the in-process mock engine without network access.
    """

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._engines: dict[str, Engine] = {}
        self._transport = transport

    @property
    def transport(self) -> httpx.AsyncBaseTransport | None:
        return self._transport

    def register(self, engine: Engine) -> None:
        self._engines[engine.model] = engine

    def resolve(self, model: str) -> Engine:
        if model not in self._engines:
            raise KeyError(f"No engine registered for model: {model!r}")
        return self._engines[model]

    def list_engines(self) -> list[Engine]:
        return list(self._engines.values())

    async def health_check(self) -> dict[str, bool]:
        """
        Query each engine's /v1/models via Contract B; an engine is healthy
        iff the request succeeds and it serves its registered model.
        """
        results: dict[str, bool] = {}
        for engine in self._engines.values():
            client = ContractBClient(engine.base_url, transport=self._transport)
            try:
                payload = await client.list_models()
                served = {m["id"] for m in payload.get("data", [])}
                ok = engine.model in served
            except (httpx.HTTPError, KeyError, TypeError):
                ok = False
            engine.healthy = ok
            engine.last_checked = datetime.now(UTC)
            results[engine.model] = ok
        return results


def default_engines(base_url: str = "http://localhost:8000", port: int = 8000) -> list[Engine]:
    """The 4-engine dev topology (placeholder pending D-03); all point at the mock engine."""
    return [Engine(model=name, base_url=base_url, port=port) for name in DEFAULT_MODELS]
