from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


class RenameSessionRequest(BaseModel):
    title: str


@router.get("")
async def list_sessions(request: Request, module: str | None = None) -> JSONResponse:
    state = request.app.state.app_state
    sessions = state.session_store.list_sessions(module=module)
    return JSONResponse(
        [
            {
                "id": s.id,
                "module": s.module,
                "title": s.title,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "message_count": s.message_count,
            }
            for s in sessions
        ]
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> JSONResponse:
    state = request.app.state.app_state
    state.session_store.delete_session(session_id)
    state.meta_agent._session_memories.pop(session_id, None)
    return JSONResponse({"status": "deleted", "session_id": session_id})


@router.patch("/{session_id}")
async def rename_session(
    session_id: str, req: RenameSessionRequest, request: Request
) -> JSONResponse:
    state = request.app.state.app_state
    state.session_store.rename_session(session_id, req.title)
    return JSONResponse({"status": "renamed", "session_id": session_id, "title": req.title})


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str, request: Request) -> JSONResponse:
    state = request.app.state.app_state
    messages = state.session_store.get_messages(session_id)
    return JSONResponse(
        [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "ts": m.ts,
                "metadata": m.metadata,
            }
            for m in messages
        ]
    )
