"""
Contract A — Band A interface definitions.

Defines the abstract interfaces that all Remanon runtime components must
implement.  No GPU or inference logic lives here — only protocols and data
types.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_DIR = Path(__file__).parent / "artifact_schemas"

# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

AgentName = str
ArtifactId = str
HBM3Handle = str | None  # opaque region identifier; None in CPU-only mode


# ---------------------------------------------------------------------------
# Artifact envelope
# ---------------------------------------------------------------------------


class Artifact:
    """Validated artifact produced by any Remanon agent."""

    def __init__(self, raw: dict[str, Any], agent_name: AgentName) -> None:
        schema = _load_schema(agent_name)
        jsonschema.validate(instance=raw, schema=schema)
        self._raw = raw

    @property
    def artifact_id(self) -> ArtifactId:
        return self._raw["artifact_id"]

    @property
    def agent(self) -> AgentName:
        return self._raw["agent"]

    @property
    def payload(self) -> dict[str, Any]:
        return self._raw["payload"]

    @property
    def hbm3_handle(self) -> HBM3Handle:
        return self._raw["payload"].get("hbm3_handle")

    def to_dict(self) -> dict[str, Any]:
        return self._raw


def _load_schema(agent_name: AgentName) -> dict[str, Any]:
    path = _SCHEMA_DIR / f"{agent_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No schema for agent '{agent_name}' at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Band A interfaces
# ---------------------------------------------------------------------------


class LeaseManager(ABC):
    """Allocates and releases named HBM3 memory regions."""

    @abstractmethod
    def acquire(self, name: str, size_bytes: int) -> HBM3Handle:
        """Reserve *size_bytes* bytes in HBM3 under *name*; return an opaque handle."""

    @abstractmethod
    def release(self, handle: HBM3Handle) -> None:
        """Return the region identified by *handle* to the pool."""

    @abstractmethod
    def list_leases(self) -> list[dict[str, Any]]:
        """Return metadata for every active lease."""


class Materializer(ABC):
    """Loads model checkpoints into leased HBM3 regions."""

    @abstractmethod
    def materialize(
        self,
        checkpoint_uri: str,
        handle: HBM3Handle,
        dtype: str = "bfloat16",
    ) -> None:
        """Load checkpoint at *checkpoint_uri* into the region at *handle*."""

    @abstractmethod
    def evict(self, handle: HBM3Handle) -> None:
        """Unload a materialized model, freeing the region contents."""


class Generator(ABC):
    """Runs inference and writes output artifacts."""

    @abstractmethod
    def generate(
        self,
        agent_name: AgentName,
        prompt: str,
        handle: HBM3Handle,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Artifact:
        """
        Run inference for *agent_name* using the model at *handle*.

        Returns a validated :class:`Artifact`.
        """
