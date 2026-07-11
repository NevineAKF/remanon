"""
Band B — generation gateway (Memory Arbiter).

The single path through which agents generate: it enforces Contract A
discipline by refusing to generate against a model whose master context has
not been materialized. Nothing bypasses lease/materialize.
"""

from __future__ import annotations

import httpx

from app.adapter.contract_b_client import ContractBClient
from core.materializer import LazyMaterializer
from core.registry import EngineRegistry, resolve_engine_transport


class NotMaterializedError(RuntimeError):
    """Generation attempted against a model with no materialized master context."""


class CoreGenerator:
    def __init__(
        self,
        registry: EngineRegistry,
        materializer: LazyMaterializer,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._registry = registry
        self._materializer = materializer
        self._transport = transport

    async def generate(
        self,
        agent_name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        if self._materializer.get_handle(model) is None:
            raise NotMaterializedError(
                f"Agent {agent_name!r}: model {model!r} has no materialized master "
                "context; call materialize() before generate()"
            )
        # WHY: every generation request carries the pinned master as its
        # byte-identical system prefix — agents genuinely read the shared
        # context, and the engine-level prefix reuse between same-model
        # agents becomes attributable to Remanon's pinned master. The prefix
        # is never None here: the guard above passed, and the materializer
        # installs handle and text atomically in one critical section.
        prefix = self._materializer.master_prefix(model)
        system_content = f"{prefix}\n\n{system_prompt}"
        engine = self._registry.resolve(model)
        # The wire request must name whatever THIS engine actually serves —
        # for a hybrid-live real engine that's its real checkpoint name
        # (e.g. "gpt-oss-20b"), not the internal placeholder `model` key.
        wire_model = engine.served_model or model
        transport = resolve_engine_transport(engine, self._transport)
        client = ContractBClient(engine.base_url, transport=transport)
        response = await client.chat_completion(
            model=wire_model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]
