from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import structlog  # type: ignore[import-untyped]
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status  # type: ignore[import-untyped]
from fastapi.responses import JSONResponse  # type: ignore[import-untyped]
from pydantic import BaseModel  # type: ignore[import-untyped]

from src.chat_contract import sanitize_output  # type: ignore[import-untyped]
from src.schemas import ChatRequest, ChatResponse  # type: ignore[import-untyped]
from src.services.chat_turns import (
    bootstrap_session,
    execute_stream_turn,
    execute_turn,
    iter_stream_chunks,
    persist_turn,
    resolve_chat_session,
)
from src.settings_store import save_partial_settings
from src.utils import safe_create_task  # type: ignore[import-untyped]

router = APIRouter(prefix="/api/v1", tags=["Chat"])
logger = structlog.get_logger("aetherforge.chat")

# Backpressure: Max 5 concurrent chat requests
CHAT_SEMAPHORE = asyncio.Semaphore(5)


def _resolve_session_id(raw_id: str, module: str) -> tuple[str, bool]:
    return resolve_chat_session(raw_id, module)


def _bootstrap_session(
    state: Any,
    session_id: str,
    module: str,
    is_new: bool,
) -> None:
    bootstrap_session(state, session_id, module, is_new)


def _strip_think_tags(content: str, is_stream: bool = False) -> str:
    """Rigorous project-wide guard to strip protocol tags and stray tool JSON."""
    return sanitize_output(content, is_stream=is_stream)


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
    if CHAT_SEMAPHORE._value == 0:  # type: ignore[attr-defined]
        logger.warning(
            "Backpressure: Chat request rejected (Semaphore full)",
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is busy. Please try again in a few seconds.",
        )

    await CHAT_SEMAPHORE.acquire()

    try:
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

        if request.module in ("ragforge", "analytics") and not request.context.get("active_docs"):
            request.context["active_docs"] = state.document_registry.get_selected_sources()

        result = await execute_turn(
            state,
            session_id=internal_sid,
            module=request.module,
            message=request.message,
            xray_mode=request.xray_mode,
            context=request.context,
        )
        await persist_turn(
            state,
            session_id=internal_sid,
            module=request.module,
            message=request.message,
            result=result,
            wait_for_persist=True,
        )
        return result
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
        {"id": "qwen-2.5-7b", "name": "Qwen 2.5 (7B Instruct)"},
        {"id": "llama-3-8b", "name": "Llama 3 (8B)"},
        {"id": "bitnet-b1.58-2b", "name": "BitNet b1.58 (2B)"},
        {"id": "mixtral-8x7b", "name": "Mixtral 8x7B"},
    ]
    
    current = getattr(state, "selected_chat_model", "qwen-2.5-7b")
    
    return JSONResponse({
        "models": models,
        "current": current
    })


class ModelSelectionPayload(BaseModel):
    model_id: str

@router.post("/chat/model")
async def update_chat_model(payload: ModelSelectionPayload, fastapi_request: Request):
    """Switch the active chat model and persist the runtime selection."""
    state = fastapi_request.app.state.app_state
    model_id = payload.model_id
    ok = await state.meta_agent.switch_model(model_id)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Could not activate chat model '{model_id}'.")
    state.selected_chat_model = state.meta_agent.model_id
    save_partial_settings({"SELECTED_CHAT_MODEL": state.meta_agent.model_id})
    logger.info("Chat Brain switched to: %s", state.meta_agent.model_id)
    return {"status": "ok", "selected": state.meta_agent.model_id}


@router.websocket("/ws/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocket streaming endpoint for real-time chat."""
    await websocket.accept()
    state = websocket.app.state.app_state

    active = True
    try:
        while active:
            try:
                data = await websocket.receive_json()
            except (WebSocketDisconnect, RuntimeError):
                logger.info("WebSocket disconnected early in loop", session_id=session_id)
                active = False
                break

            message = str(data.get("message", "")).strip()
            module = str(data.get("module", "localbuddy"))
            xray_mode = bool(data.get("xray_mode", False))
            context: dict[str, Any] = data.get("context", {}) or {}
            client_session_id = str(data.get("session_id") or session_id)

            if module in ("ragforge", "analytics") and not context.get("active_docs"):
                context["active_docs"] = state.document_registry.get_selected_sources()

            if not message:
                continue

            if CHAT_SEMAPHORE._value == 0:  # type: ignore[attr-defined]
                try:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "content": "Server is busy. Please try again in a few seconds.",
                        }
                    )
                except Exception:
                    active = False
                    break
                continue

            await CHAT_SEMAPHORE.acquire()
            try:
                # ── Explicit session lifecycle ───────────────────
                internal_sid, is_new = _resolve_session_id(client_session_id, module)
                _bootstrap_session(state, internal_sid, module, is_new)

                try:
                    await websocket.send_json(
                        {
                            "type": "meta",
                            "session_id": internal_sid,
                            "module": module,
                        }
                    )
                except Exception:
                    active = False
                    break

                # ── True Streaming Turn ──────────────────────────
                async for chunk in execute_stream_turn(
                    state,
                    session_id=internal_sid,
                    module=module,
                    message=message,
                    xray_mode=xray_mode,
                    context=context,
                ):
                    try:
                        await websocket.send_json(chunk)
                        if chunk.get("type") == "done":
                            active = False
                            break
                    except (WebSocketDisconnect, RuntimeError):
                        active = False
                        break
                
                if not active:
                    break

            except Exception as e:
                logger.error("WebSocket sub-processing error: %s", e)
                try:
                    await websocket.send_json({"type": "error", "content": f"Session Logic Error: {e}"})
                except Exception:
                    active = False
                    break
            finally:
                CHAT_SEMAPHORE.release()

    except Exception as e:
        logger.exception("WebSocket master error", session_id=session_id, error=str(e))
    finally:
        logger.info("WebSocket session terminated", session_id=session_id)
