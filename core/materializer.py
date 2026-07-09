"""
Lazy context materializer — Band B (Memory Arbiter).

materialize(context_id, model) performs the one-time master prefill through
Contract B, caches the resulting handle, and is idempotent and
concurrency-safe: N concurrent calls for the same model result in exactly
one prefill (guarded by a per-model asyncio.Lock).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from app.adapter.contract_b_client import ContractBClient
from core.registry import EngineRegistry, resolve_engine_transport


@dataclass(frozen=True, slots=True)
class MaterializedHandle:
    """Opaque reference to a materialized master context block."""

    handle_id: str
    model: str
    context_id: str
    created_at: datetime


class LazyMaterializer:
    """One master prefill per model, no matter how many agents ask."""

    def __init__(
        self,
        registry: EngineRegistry,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._registry = registry
        self._transport = transport
        self._handles: dict[str, MaterializedHandle] = {}
        # setdefault is atomic under asyncio's single-threaded scheduling
        self._locks: dict[str, asyncio.Lock] = {}
        self.prefills_performed = 0
        self.prefills_avoided = 0

    async def materialize(
        self, context_id: str, model: str, context_text: str | None = None
    ) -> MaterializedHandle:
        cached = self._handles.get(model)
        if cached is not None:
            self.prefills_avoided += 1
            return cached

        lock = self._locks.setdefault(model, asyncio.Lock())
        async with lock:
            # Double-check: another task may have prefilled while we waited.
            cached = self._handles.get(model)
            if cached is not None:
                self.prefills_avoided += 1
                return cached

            await self._prefill(context_id, model, context_text)
            handle = MaterializedHandle(
                handle_id=uuid.uuid4().hex,
                model=model,
                context_id=context_id,
                created_at=datetime.now(UTC),
            )
            self._handles[model] = handle
            self.prefills_performed += 1
            return handle

    def get_handle(self, model: str) -> MaterializedHandle | None:
        return self._handles.get(model)

    async def _prefill(self, context_id: str, model: str, context_text: str | None) -> None:
        engine = self._registry.resolve(model)
        wire_model = engine.served_model or model
        transport = resolve_engine_transport(engine, self._transport)
        client = ContractBClient(engine.base_url, transport=transport)
        content = f"[master-prefill] context_id={context_id}"
        if context_text:
            content = f"{content}\n{context_text}"
        await client.chat_completion(
            model=wire_model,
            messages=[{"role": "system", "content": content}],
            max_tokens=1,
        )
