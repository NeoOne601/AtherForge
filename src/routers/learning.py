from __future__ import annotations
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils import safe_create_task

router = APIRouter(prefix="/api/v1", tags=["Learning"])
logger = logging.getLogger(__name__)

class FeedbackPayload(BaseModel):
    session_id: str
    query: str
    response: str
    verdict: str  # "accepted" or "corrected"
    correction: str | None = None

@router.post("/learning/feedback")
async def submit_feedback(payload: FeedbackPayload, request: Request) -> JSONResponse:
    state = request.app.state.app_state
    
    metadata = {}
    if payload.correction:
        metadata["correction"] = payload.correction
        
    await state.meta_agent._sona.on_interaction(
        query=payload.query,
        response=payload.response,
        verdict=payload.verdict,
        route="user_feedback",
        metadata=metadata
    )
    return JSONResponse({"status": "recorded"})


@router.post("/learning/trigger")
async def trigger_training(request: Request) -> JSONResponse:
    from src.learning.tasks import _nightly_oplora_job

    safe_create_task(_nightly_oplora_job(request.app, force=True), name="nightly_oplora")
    return JSONResponse({"status": "triggered", "message": "OPLoRA training job started"})


@router.get("/learning/history")
async def get_learning_history(request: Request) -> JSONResponse:
    state = request.app.state.app_state
    history = state.history_manager.get_history()
    return JSONResponse(history)


@router.get("/replay/stats")
async def replay_stats(request: Request) -> JSONResponse:
    state = request.app.state.app_state
    stats = await state.replay_buffer.get_stats()
    return JSONResponse(stats)


@router.get("/replay/items")
async def replay_items(request: Request, limit: int = 50) -> JSONResponse:
    state = request.app.state.app_state
    items = await state.replay_buffer.sample(n=limit, min_faithfulness=0.0, exclude_used=False)
    items.sort(key=lambda x: x.get("timestamp_utc", 0), reverse=True)
    return JSONResponse(items)


@router.get("/learning/capacity")
async def learning_capacity(request: Request) -> JSONResponse:
    state = request.app.state.app_state
    from src.learning.oplora_manager import OPLoRAManager

    manager = OPLoRAManager(state.settings)
    manager.load_checkpoints()
    capacity = manager.estimate_capacity()
    return JSONResponse(
        {
            "capacity_pct": round(capacity * 100, 1),
            "total_tasks": sum(len(v) for v in manager._subspaces.values()),
        }
    )


@router.post("/learning/refine-text")
async def refine_text(request: Request) -> JSONResponse:
    from src.schemas import RefineRequest

    data = await request.json()
    refine_req = RefineRequest(**data)
    state = request.app.state.app_state
    refined = await state.meta_agent.refine_text(refine_req.text)
    return JSONResponse({"refined": refined})
