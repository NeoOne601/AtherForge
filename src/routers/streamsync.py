import json

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.schemas import RSSFeedRequest

router = APIRouter(prefix="/api/v1", tags=["StreamSync"])
logger = structlog.get_logger("aetherforge.streamsync")


def _load_events(limit: int | None = None) -> list[dict]:
    from src.modules.streamsync.graph import _EVENT_STREAM

    events = list(_EVENT_STREAM)
    if limit is not None and limit > 0:
        return events[-limit:]
    return events


@router.get("/events/stream")
@router.get("/streamsync/events/stream")
async def get_event_stream(limit: int | None = None) -> JSONResponse:
    return JSONResponse(_load_events(limit=limit))


@router.get("/streamsync/rss")
async def list_rss_feeds(fastapi_request: Request) -> JSONResponse:
    state = fastapi_request.app.state.app_state
    return JSONResponse({"feeds": state.streamsync_rss_feeds})


@router.post("/streamsync/rss/add")
async def add_rss_feed(request: RSSFeedRequest, fastapi_request: Request) -> JSONResponse:
    state = fastapi_request.app.state.app_state
    if request.url not in state.streamsync_rss_feeds:
        state.streamsync_rss_feeds.append(request.url)
        rss_state_file = state.settings.data_dir / "streamsync_rss_feeds.json"
        with open(rss_state_file, "w") as handle:
            json.dump({"feeds": state.streamsync_rss_feeds}, handle)
        logger.info("Added RSS feed: %s", request.url)
    return JSONResponse({"status": "Success", "feeds": state.streamsync_rss_feeds})


@router.post("/streamsync/rss/remove")
async def remove_rss_feed(request: RSSFeedRequest, fastapi_request: Request) -> JSONResponse:
    state = fastapi_request.app.state.app_state
    if request.url in state.streamsync_rss_feeds:
        state.streamsync_rss_feeds.remove(request.url)
        rss_state_file = state.settings.data_dir / "streamsync_rss_feeds.json"
        with open(rss_state_file, "w") as handle:
            json.dump({"feeds": state.streamsync_rss_feeds}, handle)
        logger.info("Removed RSS feed: %s", request.url)
    return JSONResponse({"status": "Success", "feeds": state.streamsync_rss_feeds})
