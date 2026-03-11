from __future__ import annotations
import asyncio
import structlog
import time
import uuid
from fastapi import APIRouter, Request, HTTPException, status
from src.schemas import ChatRequest, ChatResponse, RefineRequest
from src.meta_agent import MetaAgentInput
from src.utils import safe_create_task

router = APIRouter(prefix="/api/v1", tags=["Chat"])
logger = structlog.get_logger("aetherforge.chat")

# Backpressure: Max 5 concurrent chat requests
CHAT_SEMAPHORE = asyncio.Semaphore(5)

# Generic session ID prefixes sent by the UI before the user
# explicitly picks a session from history. These always get a
# fresh UUID so every new chat starts a clean conversation.
_GENERIC_PREFIXES = ("ui-session-",)


def _resolve_session_id(raw_id: str, module: str) -> tuple[str, bool]:
    """Return (internal_session_id, is_new_session).

    Rules:
      1. Generic IDs (``ui-session-*``) → generate a fresh UUID.
      2. All IDs are prefixed with ``{module}:`` so that two modules
         can never share the same session row in the DB.
    """
    is_new = any(raw_id.startswith(p) for p in _GENERIC_PREFIXES)
    base = str(uuid.uuid4()) if is_new else raw_id

    # Ensure module prefix is present (idempotent for history IDs
    # that are already prefixed).
    if not base.startswith(f"{module}:"):
        internal = f"{module}:{base}"
    else:
        internal = base

    return internal, is_new


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
        session_store = state.session_store

        # ── Resolve session ID ────────────────────────────────
        internal_sid, is_new = _resolve_session_id(
            request.session_id, request.module,
        )
        logger.debug(
            "Session resolved",
            raw=request.session_id,
            internal=internal_sid,
            is_new=is_new,
        )

        # ── Session bootstrap ─────────────────────────────────
        if is_new or not session_store.session_exists(internal_sid):
            session_store.create_session(
                module=request.module,
                session_id=internal_sid,
            )
        elif internal_sid not in state.meta_agent._session_memories:
            try:
                restored = session_store.to_langchain_messages(
                    internal_sid,
                )
                if restored:
                    state.meta_agent._session_memories[
                        internal_sid
                    ] = restored
            except Exception as e:
                logger.warning("Session hydration failed: %s", e)

        result = await state.meta_agent.run(
            MetaAgentInput(
                session_id=internal_sid,
                module=request.module,
                message=request.message,
                xray_mode=request.xray_mode,
                context=request.context,
                system_location=state.system_location,
            )
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist (non-blocking)
        sid_for_persist = internal_sid

        def _persist() -> None:
            meta = {
                "module": request.module,
                "latency_ms": round(latency_ms, 2),
                "faithfulness_score": result.faithfulness_score,
            }
            session_store.append_message(
                sid_for_persist, "user", request.message,
            )
            session_store.append_message(
                sid_for_persist, "assistant", result.response, meta,
            )

        safe_create_task(
            asyncio.to_thread(_persist), name="persist_session",
        )

        # Record to replay (non-blocking)
        safe_create_task(
            state.replay_buffer.record(
                session_id=internal_sid,
                module=request.module,
                prompt=request.message,
                response=result.response,
                tool_calls=result.tool_calls,
                faithfulness_score=result.faithfulness_score,
            )
        )

        return ChatResponse(
            session_id=internal_sid,
            response=result.response,
            module=request.module,
            latency_ms=round(latency_ms, 2),
            tool_calls=result.tool_calls,
            policy_decisions=result.policy_decisions,
            causal_graph=(
                result.causal_graph if request.xray_mode else None
            ),
            faithfulness_score=result.faithfulness_score,
        )
    except Exception as exc:
        logger.exception("Chat error: %s", exc)
        raise HTTPException(
            status_code=500, detail=str(exc),
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
