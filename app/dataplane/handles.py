"""Band C — zero-copy tensor handle passing between agents (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorHandle:
    """
    Opaque reference to a tensor region in HBM3 (or host RAM in CPU mode).

    Agents pass these handles instead of copying tensor data, achieving
    zero-copy inter-agent communication once the GPU path is implemented.
    """

    handle_id: str
    agent_owner: str
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    resident: bool = False  # True when physically in HBM3

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle_id": self.handle_id,
            "agent_owner": self.agent_owner,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "size_bytes": self.size_bytes,
            "resident": self.resident,
        }
