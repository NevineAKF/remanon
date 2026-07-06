"""Band C — Contract B adapter: translates internal calls to the OpenAI-compatible vLLM API."""

from __future__ import annotations

import httpx


class ContractBClient:
    """
    Thin async wrapper around the OpenAI-compatible inference endpoint.

    An optional httpx transport can be injected so tests can route requests
    to an in-process ASGI app (no network).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    def _make_client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict:
        async with self._make_client(60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            return response.json()

    async def list_models(self) -> dict:
        async with self._make_client(10.0) as client:
            response = await client.get(f"{self._base_url}/v1/models")
            response.raise_for_status()
            return response.json()
