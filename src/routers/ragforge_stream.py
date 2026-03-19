# AetherForge v1.0 — src/routers/ragforge_stream.py
# ─────────────────────────────────────────────────────────────────
# Progressive RAG — Citation Streaming Endpoint (Phase 17)
#
# Exposes CognitiveRAG.think_and_answer_stream() via Server-Sent Events.
# Frontend receives real-time tokens with citation attribution — each
# sentence is tagged with the document chunk that grounds it.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/ragforge", tags=["RAGForge Stream"])
logger = structlog.get_logger("aetherforge.ragforge_stream")


class StreamQueryRequest(BaseModel):
    query: str
    source_filter: str | list[str] | None = None
    prompt_variant: str = "v1"


@router.post("/stream")
async def stream_rag_answer(payload: StreamQueryRequest, request: Request) -> StreamingResponse:
    """
    Progressive RAG streaming endpoint.

    Returns a Server-Sent Events (SSE) stream where each event is a JSON object:

        event: reasoning   — thinking trace tokens (not shown to the user)
        event: citation    — citation registration (source, page, excerpt)
        event: token       — answer words with optional citation_id
        event: done        — pipeline summary (latency, evidence count, etc.)

    Frontend usage:
        const es = new EventSource('/api/v1/ragforge/stream', { method: 'POST' });
        es.onmessage = (e) => handleEvent(JSON.parse(e.data));
    """
    state = request.app.state.app_state
    rag: any = getattr(state, "cognitive_rag", None)

    if not rag:
        async def error_stream():
            yield "data: " + json.dumps({"type": "error", "content": "CognitiveRAG not initialized."}) + "\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    active_docs = getattr(state, "active_docs", [])
    source_filter = payload.source_filter or (active_docs[0] if len(active_docs) == 1 else active_docs or None)

    async def event_generator():
        try:
            async for event in rag.think_and_answer_stream(
                query=payload.query,
                source_filter=source_filter,
                prompt_variant=payload.prompt_variant,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("RAG stream error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
