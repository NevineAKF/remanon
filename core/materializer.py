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


def master_system_content(context_id: str, context_text: str | None) -> str:
    """Build the master system prefix for a context.

    This is the SINGLE SOURCE OF TRUTH for the shared prefix: both the
    prefill request and every generation request build their system content
    from it, so the two wire paths can never drift apart byte-wise.
    Byte-identity of this prefix is what makes same-model agents share the
    master's token prefix at the engine level.
    """
    return f"MASTER CONTEXT [{context_id}]\n{context_text or ''}"


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
        self._texts: dict[str, str] = {}
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
            # WHY: the master text and the handle must become visible
            # atomically together (same critical section), so a model with a
            # handle always has the text its wire prefix is rebuilt from.
            self._texts[model] = context_text or ""
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

    def master_text(self, model: str) -> str | None:
        """The master context text stored at prefill time, or None if not materialized."""
        return self._texts.get(model)

    def master_prefix(self, model: str) -> str | None:
        """The exact wire system prefix for `model`, or None if not materialized."""
        handle = self._handles.get(model)
        if handle is None:
            return None
        return master_system_content(handle.context_id, self._texts.get(model))

    async def _prefill(self, context_id: str, model: str, context_text: str | None) -> None:
        engine = self._registry.resolve(model)
        wire_model = engine.served_model or model
        transport = resolve_engine_transport(engine, self._transport)
        client = ContractBClient(engine.base_url, transport=transport)
        # WHY: the prefill seeds the engine cache with exactly the prefix
        # agents will send — same single-source-of-truth builder, so the
        # cached block and every generation request match byte-for-byte.
        content = master_system_content(context_id, context_text)
        await client.chat_completion(
            model=wire_model,
            # A system-only conversation isn't generation-ready for chat
            # templates that require alternating/user-terminated roles (real
            # vLLM servers 400 on it) — a minimal user turn makes it valid
            # without changing what this call does (max_tokens=1: no real
            # output is needed, this is purely a prefill to pin the master
            # block into KV cache).
            messages=[{"role": "system", "content": content}, {"role": "user", "content": "."}],
            max_tokens=1,
        )
