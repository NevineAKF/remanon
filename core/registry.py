"""Band B — central catalog of live artifacts and agent handles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.contract_a import AgentName, Artifact, ArtifactId


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
