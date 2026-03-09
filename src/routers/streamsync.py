import json
import asyncio
import structlog
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from src.schemas import RSSFeedRequest

router = APIRouter(prefix="/api/v1/streamsync", tags=["StreamSync"])
logger = structlog.get_logger("aetherforge.streamsync")

@router.get("/events/stream")
async def get_event_stream() -> JSONResponse:
    from src.modules.streamsync.graph import _EVENT_STREAM
    return JSONResponse(list(_EVENT_STREAM))

@router.post("/rss/add")
async def add_rss_feed(request: RSSFeedRequest, fastapi_request: Request) -> JSONResponse:
    state = fastapi_request.app.state.app_state
    if request.url not in state.streamsync_rss_feeds:
        state.streamsync_rss_feeds.append(request.url)
        # Persist
        rss_state_file = state.settings.data_dir / "streamsync_rss_feeds.json"
        with open(rss_state_file, "w") as f:
            json.dump({"feeds": state.streamsync_rss_feeds}, f)
        logger.info("Added RSS feed: %s", request.url)
    return JSONResponse({"status": "Success", "feeds": state.streamsync_rss_feeds})

@router.post("/rss/remove")
async def remove_rss_feed(request: RSSFeedRequest, fastapi_request: Request) -> JSONResponse:
    state = fastapi_request.app.state.app_state
    if request.url in state.streamsync_rss_feeds:
        state.streamsync_rss_feeds.remove(request.url)
        # Persist
        rss_state_file = state.settings.data_dir / "streamsync_rss_feeds.json"
        with open(rss_state_file, "w") as f:
            json.dump({"feeds": state.streamsync_rss_feeds}, f)
        logger.info("Removed RSS feed: %s", request.url)
    return JSONResponse({"status": "Success", "feeds": state.streamsync_rss_feeds})
