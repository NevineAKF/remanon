"""
Band A, Layer L7 — agent base.

run(case) = lease (Contract A) → materialize → generate through the core →
parse → schema-validated Artifact. The lease is always released, even on
failure. Parsing tolerates code fences and surrounding whitespace.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from app.adapter.prompts import PROMPTS
from app.dataplane.store import TelemetryStore
from contracts.contract_a import AgentName, Artifact
from core.generator import CoreGenerator
from core.materializer import LazyMaterializer
from core.memory_model import AGENT_MODEL_MAP
from core.residency import ResidencyManager

MASTER_CONTEXT_ID = "master-digest"

_FENCE_OPEN = re.compile(r"^```[\w-]*\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")


def parse_json_payload(text: str) -> dict[str, Any]:
    """Parse model output into a dict; strips code fences, tolerates whitespace."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", candidate)).strip()
    obj = json.loads(candidate)  # JSONDecodeError is a ValueError
    if not isinstance(obj, dict):
        raise ValueError(f"Model output is not a JSON object: {type(obj).__name__}")
    return obj


class BaseAgent:
    name: AgentName = ""

    def __init__(
        self,
        residency: ResidencyManager,
        materializer: LazyMaterializer,
        generator: CoreGenerator,
        store: TelemetryStore | None = None,
    ) -> None:
        self._residency = residency
        self._materializer = materializer
        self._generator = generator
        self._store = store

    @property
    def model(self) -> str:
        return AGENT_MODEL_MAP[self.name]

    async def run(self, case: dict[str, Any]) -> Artifact:
        model = self.model
        lease = await self._residency.lease(MASTER_CONTEXT_ID, model, self.name)
        try:
            await self._materializer.materialize(MASTER_CONTEXT_ID, model)
            text = await self._generator.generate(
                self.name,
                model,
                self.system_prompt(),
                self.user_prompt(case),
            )
            payload = parse_json_payload(text)
            return self._wrap(payload)
        finally:
            await self._residency.release(lease.lease_id)

    def system_prompt(self) -> str:
        return PROMPTS[self.name]

    def user_prompt(self, case: dict[str, Any]) -> str:
        return "CASE:\n" + json.dumps(case, default=str)

    def _wrap(self, payload: dict[str, Any]) -> Artifact:
        raw = {
            "artifact_id": str(uuid.uuid4()),
            "agent": self.name,
            "version": "0.1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        return Artifact(raw, self.name)  # validates against the role's JSON schema
