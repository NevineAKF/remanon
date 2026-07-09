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


# Sentinel distinguishing "this Engine has no per-engine transport override"
# from a real, meaningful value of None (None means genuine real-network TCP,
# so it can't double as "unset"). Every Engine built before hybrid-live
# existed never set .transport, so it's always _UNSET for them — behavior
# for every existing caller is byte-for-byte unchanged.
_UNSET: Any = object()


def resolve_engine_transport(
    engine: Engine, default: httpx.AsyncBaseTransport | None
) -> httpx.AsyncBaseTransport | None:
    """
    An engine's own .transport overrides the registry/generator-wide
    default — this is what lets one run mix a real engine (transport=None,
    real TCP) with in-process mocks (transport=ASGITransport) at the same
    time. Falls back to `default` when the engine never set one.
    """
    return default if engine.transport is _UNSET else engine.transport


@dataclass
class Engine:
    """One inference engine serving exactly one model via Contract B."""

    model: str
    base_url: str
    port: int
    # The model name THIS engine actually answers to on the wire (Contract B
    # request body). None means "same as `model`" — true for every mock
    # engine and for a real engine that happens to serve the placeholder
    # name directly. Set explicitly when a real engine only recognizes its
    # own real checkpoint name (e.g. registry key "remanon-triage-7b" but
    # the real vLLM server only serves "gpt-oss-20b").
    served_model: str | None = None
    # Per-engine transport override — see resolve_engine_transport(). Left
    # at _UNSET, an engine defers to whatever transport its generator/
    # materializer/registry was given, exactly as before this field existed.
    transport: Any = _UNSET
    healthy: bool | None = None  # None = never checked
    last_checked: datetime | None = None


class EngineRegistry:
    """
    Maps model name → engine endpoint and tracks health via Contract B.

    An optional httpx transport is threaded through to the Contract B client
    so tests can target the in-process mock engine without network access.
    Per-engine transports (see Engine.transport) override this default,
    which is what makes a hybrid real+mock registry possible.
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
        iff the request succeeds and it serves its registered (real) model.
        """
        results: dict[str, bool] = {}
        for engine in self._engines.values():
            transport = resolve_engine_transport(engine, self._transport)
            client = ContractBClient(engine.base_url, transport=transport)
            wire_model = engine.served_model or engine.model
            try:
                payload = await client.list_models()
                served = {m["id"] for m in payload.get("data", [])}
                ok = wire_model in served
            except (httpx.HTTPError, KeyError, TypeError):
                ok = False
            engine.healthy = ok
            engine.last_checked = datetime.now(UTC)
            results[engine.model] = ok
        return results


def default_engines(base_url: str = "http://localhost:8000", port: int = 8000) -> list[Engine]:
    """The 4-engine dev topology (placeholder pending D-03); all point at the mock engine."""
    return [Engine(model=name, base_url=base_url, port=port) for name in DEFAULT_MODELS]
