from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from src.chat_contract import build_visible_reasoning_trace, resolve_session_id, split_reasoning_trace
from src.meta_agent import MetaAgentInput
from src.schemas import ChatRequest, ChatResponse
from src.utils import safe_create_task

router = APIRouter(prefix="/api/v1", tags=["Chat"])
logger = structlog.get_logger("aetherforge.chat")

# Backpressure: Max 5 concurrent chat requests
CHAT_SEMAPHORE = asyncio.Semaphore(5)

def _resolve_session_id(raw_id: str, module: str) -> tuple[str, bool]:
    """Return (internal_session_id, is_new_session).

    Rules:
      1. Generic IDs (``ui-session-*``) → generate a fresh UUID.
      2. All IDs are prefixed with ``{module}:`` so that two modules
         can never share the same session row in the DB.
    """
    return resolve_session_id(raw_id, module)


def _bootstrap_session(
    state: Any,
    session_id: str,
    module: str,
    is_new: bool,
) -> None:
    """Ensure the durable session row and in-memory history both exist."""
    session_store = state.session_store

    if is_new or not session_store.session_exists(session_id):
        session_store.create_session(module=module, session_id=session_id)
        return

    if session_id in state.meta_agent._session_memories:
        return

    try:
        restored = session_store.to_langchain_messages(session_id)
        if restored:
            state.meta_agent._session_memories[session_id] = restored
    except Exception as exc:
        logger.warning("Session hydration failed", session_id=session_id, error=str(exc))


async def _run_protocol(
    state: Any,
    *,
    session_id: str,
    module: str,
    message: str,
    xray_mode: bool,
    context: dict[str, Any],
    protocol: str,
) -> ChatResponse:
    """Execute the canonical MetaAgent orchestration path."""
    # We ignore the 'protocol' parameter and always use MetaAgent which is now canonical
    result = await state.meta_agent.run(
        MetaAgentInput(
            session_id=session_id,
            module=module,
            message=message,
            xray_mode=xray_mode,
            context=context,
            system_location=state.system_location,
        )
    )

    return ChatResponse(
        session_id=session_id,
        response=result.response,
        module=module,
        latency_ms=0.0,
        tool_calls=result.tool_calls,
        policy_decisions=result.policy_decisions,
        causal_graph=(result.causal_graph if xray_mode else None),
        faithfulness_score=result.faithfulness_score,
        citations=result.citations,
        attachments=result.attachments,
    )


def _schedule_turn_side_effects(
    state: Any,
    *,
    session_id: str,
    module: str,
    message: str,
    result: ChatResponse,
) -> None:
    """Persist the turn and enqueue replay recording without blocking the request."""

    def _persist() -> None:
        reasoning_trace, answer_text = _split_reasoning_trace(result.response)
        meta = {
            "module": result.module,
            "latency_ms": result.latency_ms,
            "faithfulness_score": result.faithfulness_score,
            "reasoning_trace": reasoning_trace or result.reasoning_trace,
            "answer_text": answer_text or result.answer_text,
            "citations": result.citations,
            "attachments": result.attachments,
        }
        state.session_store.append_message(session_id, "user", message)
        state.session_store.append_message(session_id, "assistant", result.response, meta)

    safe_create_task(asyncio.to_thread(_persist), name="persist_session")
    safe_create_task(
        state.replay_buffer.record(
            session_id=session_id,
            module=module,
            prompt=message,
            response=result.response,
            tool_calls=result.tool_calls,
            faithfulness_score=result.faithfulness_score,
        ),
        name="record_replay",
    )


def _iter_stream_chunks(text: str, chunk_size: int = 32) -> list[str]:
    """Break a completed response into small chunks for the UI stream."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _split_reasoning_trace(text: str) -> tuple[str | None, str]:
    """Extract a visible reasoning block from <think>...</think> output."""
    return split_reasoning_trace(text)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    fastapi_request: Request,
) -> ChatResponse:
    """Main chat endpoint (modularized).

    Routes message through MetaAgent with 429 backpressure.
    Session IDs are module-scoped to guarantee isolation.
    """
    state = fastapi_request.app.state.app_state

    # Backpressure check
    if CHAT_SEMAPHORE._value == 0:
        logger.warning(
            "Backpressure: Chat request rejected (Semaphore full)",
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is busy. Please try again in a few seconds.",
        )

    await CHAT_SEMAPHORE.acquire()

    try:
        t0 = time.perf_counter()

        # ── Resolve session ID ────────────────────────────────
        internal_sid, is_new = _resolve_session_id(
            request.session_id,
            request.module,
        )
        logger.debug(
            "Session resolved",
            raw=request.session_id,
            internal=internal_sid,
            is_new=is_new,
        )

        # ── Session bootstrap ─────────────────────────────────
        _bootstrap_session(state, internal_sid, request.module, is_new)

        result = await _run_protocol(
            state,
            session_id=internal_sid,
            module=request.module,
            message=request.message,
            xray_mode=request.xray_mode,
            context=request.context,
            protocol=request.protocol,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        reasoning_trace, answer_text = _split_reasoning_trace(result.response)
        reasoning_trace = build_visible_reasoning_trace(
            module=result.module,
            message=request.message,
            answer_text=answer_text,
            tool_calls=result.tool_calls,
            citations=result.citations,
            existing=reasoning_trace,
        )
        final_result = ChatResponse(
            session_id=internal_sid,
            response=result.response,
            module=result.module,
            latency_ms=round(latency_ms, 2),
            tool_calls=result.tool_calls,
            policy_decisions=result.policy_decisions,
            causal_graph=result.causal_graph,
            faithfulness_score=result.faithfulness_score,
            reasoning_trace=reasoning_trace,
            answer_text=answer_text,
            citations=result.citations,
            attachments=result.attachments,
        )
        _schedule_turn_side_effects(
            state,
            session_id=internal_sid,
            module=request.module,
            message=request.message,
            result=final_result,
        )
        return final_result
    except Exception as exc:
        logger.exception("Chat error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
    finally:
        CHAT_SEMAPHORE.release()


@router.get("/chat-models")
async def get_chat_models(fastapi_request: Request):
    """Return available local chat models."""
    state = fastapi_request.app.state.app_state
    models = [
        {"id": "bitnet-b1.58-2b", "name": "BitNet b1.58 (2B)"},
        {"id": "gemma-2b", "name": "Gemma 1.1 (2B)"},
        {"id": "llama-3-8b", "name": "Llama 3 (8B)"},
    ]
    current = getattr(state, "selected_chat_model", "bitnet-b1.58-2b")
    return {"models": models, "selected": current}


@router.post("/chat-model-select")
async def select_chat_model(
    payload: dict,
    fastapi_request: Request,
):
    """Store the selected model in app state for MetaAgent to use."""
    state = fastapi_request.app.state.app_state
    model_id = payload.get("model_id", "bitnet-b1.58-2b")
    state.selected_chat_model = model_id
    logger.info("Chat Brain switched to: %s", model_id)
    return {"status": "ok", "selected": model_id}


@router.websocket("/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocket streaming endpoint for real-time chat."""
    await websocket.accept()
    state = websocket.app.state.app_state

    try:
        while True:
            data = await websocket.receive_json()
            message = str(data.get("message", "")).strip()
            module = str(data.get("module", "localbuddy"))
            protocol = str(data.get("protocol", "legacy"))
            xray_mode = bool(data.get("xray_mode", False))
            context = data.get("context", {}) or {}
            client_session_id = str(data.get("session_id") or session_id)

            if not message:
                continue

            if CHAT_SEMAPHORE._value == 0:
                await websocket.send_json(
                    {
                        "type": "error",
                        "content": "Server is busy. Please try again in a few seconds.",
                    }
                )
                continue

            await CHAT_SEMAPHORE.acquire()
            try:
                t0 = time.perf_counter()
                internal_sid, is_new = _resolve_session_id(client_session_id, module)
                _bootstrap_session(state, internal_sid, module, is_new)

                await websocket.send_json(
                    {
                        "type": "meta",
                        "session_id": internal_sid,
                        "module": module,
                    }
                )

                result = await _run_protocol(
                    state,
                    session_id=internal_sid,
                    module=module,
                    message=message,
                    xray_mode=xray_mode,
                    context=context,
                    protocol=protocol,
                )
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                reasoning_trace, answer_text = _split_reasoning_trace(result.response)
                reasoning_trace = build_visible_reasoning_trace(
                    module=result.module,
                    message=message,
                    answer_text=answer_text,
                    tool_calls=result.tool_calls,
                    citations=result.citations,
                    existing=reasoning_trace,
                )
                final_result = ChatResponse(
                    session_id=internal_sid,
                    response=result.response,
                    module=result.module,
                    latency_ms=latency_ms,
                    tool_calls=result.tool_calls,
                    policy_decisions=result.policy_decisions,
                    causal_graph=result.causal_graph,
                    faithfulness_score=result.faithfulness_score,
                    reasoning_trace=reasoning_trace,
                    answer_text=answer_text,
                    citations=result.citations,
                    attachments=result.attachments,
                )

                _schedule_turn_side_effects(
                    state,
                    session_id=internal_sid,
                    module=module,
                    message=message,
                    result=final_result,
                )

                if final_result.reasoning_trace:
                    await websocket.send_json(
                        {"type": "reasoning", "content": final_result.reasoning_trace}
                    )

                stream_text = final_result.answer_text or final_result.response
                for chunk in _iter_stream_chunks(stream_text):
                    await websocket.send_json({"type": "token", "content": chunk})
                    await asyncio.sleep(0)

                await websocket.send_json(
                    {
                        "type": "done",
                        "session_id": internal_sid,
                        "module": final_result.module,
                        "latency_ms": final_result.latency_ms,
                        "faithfulness_score": final_result.faithfulness_score,
                        "policy_decisions": final_result.policy_decisions,
                        "causal_graph": final_result.causal_graph,
                        "tool_calls": final_result.tool_calls,
                        "reasoning_trace": final_result.reasoning_trace,
                        "answer_text": final_result.answer_text,
                        "citations": final_result.citations,
                        "attachments": final_result.attachments,
                    }
                )
            finally:
                CHAT_SEMAPHORE.release()

    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected", session_id=session_id)
    except Exception as e:
        logger.exception("Chat WebSocket error", session_id=session_id, error=str(e))
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass
