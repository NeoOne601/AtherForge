from __future__ import annotations
import asyncio
import structlog
from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import Response as FastAPIResponse

router = APIRouter(prefix="/api/v1/sessions", tags=["Export"])
logger = structlog.get_logger("aetherforge.export")

@router.get("/{session_id}/export")
async def export_session(
    session_id: str,
    request: Request,
    format: str = "md",
    message_id: str | None = None,
) -> FastAPIResponse:
    """
    Export a full session (or a single message) as Markdown or PDF.
    """
    state = request.app.state.app_state
    engine = state.export_engine
    safe_id = session_id[:8]

    try:
        if format == "pdf":
            if message_id:
                content = await asyncio.to_thread(engine.message_to_pdf, session_id, message_id)
                filename = f"aetherforge_response_{safe_id}.pdf"
            else:
                content = await asyncio.to_thread(engine.session_to_pdf, session_id)
                filename = f"aetherforge_session_{safe_id}.pdf"
            return FastAPIResponse(
                content=content,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            if message_id:
                content = engine.message_to_markdown(session_id, message_id)
                filename = f"aetherforge_response_{safe_id}.md"
            else:
                content = engine.session_to_markdown(session_id)
                filename = f"aetherforge_session_{safe_id}.md"
            return FastAPIResponse(
                content=content.encode("utf-8"),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
    except Exception as e:
        logger.error("Export failed for session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
