from __future__ import annotations
import asyncio
import structlog
import time
from fastapi import APIRouter, Request, HTTPException, status
from src.schemas import ChatRequest, ChatResponse, RefineRequest
from src.meta_agent import MetaAgentInput
from src.utils import safe_create_task

router = APIRouter(prefix="/api/v1", tags=["Chat"])
logger = structlog.get_logger("aetherforge.chat")

# Backpressure: Max 5 concurrent chat requests
CHAT_SEMAPHORE = asyncio.Semaphore(5)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    """
    Main chat endpoint (modularized).
    Routes message through MetaAgent with 429 backpressure.
    """
    state = fastapi_request.app.state.app_state
    
    # Backpressure check: Immediate rejection if more than 5 concurrent requests
    if CHAT_SEMAPHORE._value == 0:
        logger.warning("Backpressure: Chat request rejected (Semaphore full)")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is busy. Please try again in a few seconds."
        )
    
    await CHAT_SEMAPHORE.acquire()

    try:
        t0 = time.perf_counter()
        session_store = state.session_store

        # Session bootstrap
        if not session_store.session_exists(request.session_id):
            session_store.create_session(module=request.module, session_id=request.session_id)
        elif request.session_id not in state.meta_agent._session_memories:
            try:
                restored = session_store.to_langchain_messages(request.session_id)
                if restored:
                    state.meta_agent._session_memories[request.session_id] = restored
            except Exception as e:
                logger.warning("Session hydration failed: %s", e)

        result = await state.meta_agent.run(
            MetaAgentInput(
                session_id=request.session_id,
                module=request.module,
                message=request.message,
                xray_mode=request.xray_mode,
                context=request.context,
                system_location=state.system_location,
            )
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist (non-blocking)
        def _persist():
            meta = {"module": request.module, "latency_ms": round(latency_ms, 2), "faithfulness_score": result.faithfulness_score}
            session_store.append_message(request.session_id, "user", request.message)
            session_store.append_message(request.session_id, "assistant", result.response, meta)
        
        safe_create_task(asyncio.to_thread(_persist), name="persist_session")

        # Record to replay (non-blocking)
        safe_create_task(
            state.replay_buffer.record(
                session_id=request.session_id,
                module=request.module,
                prompt=request.message,
                response=result.response,
                tool_calls=result.tool_calls,
                faithfulness_score=result.faithfulness_score,
            )
        )

        return ChatResponse(
            session_id=request.session_id,
            response=result.response,
            module=request.module,
            latency_ms=round(latency_ms, 2),
            tool_calls=result.tool_calls,
            policy_decisions=result.policy_decisions,
            causal_graph=result.causal_graph if request.xray_mode else None,
            faithfulness_score=result.faithfulness_score,
        )
    except Exception as exc:
        logger.exception("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        CHAT_SEMAPHORE.release()

@router.get("/chat-models")
async def get_chat_models(fastapi_request: Request):
    # Retrieve models (hardcoded local models)
    models = [
        {"id": "bitnet-b1.58-2b", "name": "BitNet b1.58 (2B)"},
        {"id": "gemma-2b", "name": "Gemma 1.1 (2B)"},
        {"id": "llama-3-8b", "name": "Llama 3 (8B)"}
    ]
    return {"models": models, "selected": "bitnet-b1.58-2b"}

@router.post("/chat-model-select")
async def select_chat_model(payload: dict, fastapi_request: Request):
    """Update preferred model for the UI."""
    return {"status": "ok", "selected": payload.get("model_id")}
