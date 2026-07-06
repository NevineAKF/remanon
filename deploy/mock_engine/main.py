"""
Mock inference engine — Contract B stub.

Imitates the OpenAI-compatible vLLM REST API so that all Remanon components
can be developed and tested without a physical GPU or real model weights.

Endpoints implemented:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions
"""

from __future__ import annotations

import json
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Remanon Mock Engine",
    description="Contract B stub — OpenAI-compatible vLLM imitation",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Static model catalogue
# ---------------------------------------------------------------------------

_MODELS = [
    {
        "id": "remanon-triage-7b",
        "object": "model",
        "created": 1_700_000_000,
        "owned_by": "remanon",
    },
    {
        "id": "remanon-correlator-13b",
        "object": "model",
        "created": 1_700_000_000,
        "owned_by": "remanon",
    },
    {
        "id": "remanon-hunter-13b",
        "object": "model",
        "created": 1_700_000_000,
        "owned_by": "remanon",
    },
    {
        "id": "remanon-topology-7b",
        "object": "model",
        "created": 1_700_000_000,
        "owned_by": "remanon",
    },
    {
        "id": "remanon-reporter-13b",
        "object": "model",
        "created": 1_700_000_000,
        "owned_by": "remanon",
    },
]

_MODEL_IDS = {m["id"] for m in _MODELS}

# ---------------------------------------------------------------------------
# Deterministic role payloads (Contract B ← Layer L6 role markers)
#
# When a request's prompt contains [ROLE:name], the mock answers with a
# fixed, schema-valid JSON payload for that role so agents can parse real
# structure without a GPU.
# ---------------------------------------------------------------------------

_ROLE_RE = re.compile(r"\[ROLE:([a-z]+)\]")

_ROLE_PAYLOADS: dict[str, dict] = {
    "triage": {
        "severity": "high",
        "category": "storage-io",
        "summary": "Burst of WARN/ERROR datanode events indicates degraded block transfers.",
        "routing": ["correlator", "hunter", "topology"],
        "confidence": 0.86,
        "hbm3_handle": None,
    },
    "correlator": {
        "clusters": [
            {
                "cluster_id": "c-1",
                "members": ["blk_-1608999687919862906", "blk_8229193803249955061"],
                "score": 0.83,
                "label": "broken-pipe-transfers",
            }
        ],
        "cross_references": [],
        "hbm3_handle": None,
    },
    "hunter": {
        "findings": [
            {
                "finding_id": "f-1",
                "title": "Repeated block transfer failures across datanodes",
                "severity": "high",
                "evidence": [
                    "java.io.IOException: Broken pipe",
                    "java.io.IOException: Connection reset by peer",
                ],
                "mitre_technique": None,
                "confidence": 0.78,
            }
        ],
        "search_depth": 2,
        "hbm3_handle": None,
    },
    "topology": {
        "nodes": [
            {
                "node_id": "10.250.19.102",
                "kind": "host",
                "label": "datanode-10.250.19.102",
                "attributes": {},
            },
            {
                "node_id": "10.251.73.220",
                "kind": "host",
                "label": "datanode-10.251.73.220",
                "attributes": {},
            },
        ],
        "edges": [
            {
                "source": "10.250.19.102",
                "target": "10.251.73.220",
                "relation": "block_transfer",
                "weight": 1.0,
            }
        ],
        "hbm3_handle": None,
    },
    "reporter": {
        "title": "RCA: degraded HDFS block replication",
        "executive_summary": (
            "A burst of WARN/ERROR events shows repeated block-transfer failures "
            "between datanodes. Root cause: network instability on the "
            "inter-datanode link causing broken pipes and connection resets."
        ),
        "sections": [
            {
                "heading": "Findings",
                "body": "Repeated broken-pipe failures during block transfer.",
                "severity": "high",
            },
            {
                "heading": "Recommendation",
                "body": "Inspect the NIC and switch link between the affected datanodes.",
                "severity": None,
            },
        ],
        "source_artifacts": [],
        "overall_severity": "high",
        "hbm3_handle": None,
    },
}


def _detect_role(messages: list[ChatMessage]) -> str | None:
    for msg in messages:
        match = _ROLE_RE.search(msg.content)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI-compatible subset)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine": "mock", "gpu": False}


@app.get("/v1/models")
async def list_models() -> dict:
    return {"object": "list", "data": _MODELS}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict:
    if req.model not in _MODEL_IDS:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found.")

    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported by mock engine.")

    role = _detect_role(req.messages)
    if role in _ROLE_PAYLOADS:
        mock_text = json.dumps(_ROLE_PAYLOADS[role])
    else:
        last_user_msg = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            "(no user message)",
        )
        mock_text = (
            f"[MOCK ENGINE] Model={req.model} | "
            f"prompt_preview={last_user_msg[:80]!r} | "
            f"max_tokens={req.max_tokens} | temperature={req.temperature}"
        )
    prompt_tokens = sum(len(m.content.split()) for m in req.messages)
    completion_tokens = len(mock_text.split())

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": mock_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Global error handler — return OpenAI-style error envelopes
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_error",
                "code": 500,
            }
        },
    )
