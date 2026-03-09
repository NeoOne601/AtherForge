# AetherForge v1.0 — src/modules/streamsync/tools.py
import json
import structlog
from collections import Counter
from typing import Any

from src.modules.streamsync.graph import _EVENT_STREAM

logger = structlog.get_logger("aetherforge.streamsync.tools")

def get_tools() -> list[dict[str, Any]]:
    """Return StreamSync-specific LLM tool definitions."""
    return [
        {
            "name": "query_stream",
            "description": (
                "Fetch the most recent events from the StreamSync live buffer. "
                "CALL THIS IMMEDIATELY when the user asks to see, show, analyze, or process "
                "event streams, recent events, or live data. "
                "Do not explain how to use it — just call it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent events to retrieve (default 10, max 50)"
                    },
                    "source_filter": {
                        "type": "string",
                        "description": "Optional: filter events by source name (e.g., 'github', 'WatchTower UI')"
                    }
                },
                "required": ["limit"]
            }
        },
        {
            "name": "summarize_stream",
            "description": (
                "Summarize the StreamSync buffer: total events, events per source, "
                "most active sources, and time range of data. "
                "CALL THIS when the user asks what's in the stream, what is most active, "
                "what patterns exist, or for an overview of the event data."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "clear_buffer",
            "description": (
                "Clear all events from the StreamSync ring buffer. "
                "Call this when the user explicitly asks to clear, reset, or flush the stream."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]

def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a StreamSync tool and return string result."""
    logger.info("StreamSync Tool Execution: %s(%s)", name, args)

    if name == "query_stream":
        limit = min(int(args.get("limit", 10)), 50)
        source = args.get("source_filter")

        events = list(_EVENT_STREAM)
        if source:
            events = [e for e in events if e.get("source") == source]

        results = events[-limit:]
        if not results:
            return json.dumps({
                "count": 0,
                "events": [],
                "message": "No events in the StreamSync buffer yet. Fire events at POST /api/v1/events to populate it."
            })
        return json.dumps({"count": len(results), "events": results}, default=str)

    elif name == "summarize_stream":
        events = list(_EVENT_STREAM)
        if not events:
            return json.dumps({
                "total_events": 0,
                "message": "StreamSync buffer is empty. No events have been ingested yet."
            })

        source_counts = Counter(e.get("source", "unknown") for e in events)
        top_sources = source_counts.most_common(5)

        # Get time range if timestamps available
        timestamps = [e.get("ts") or e.get("timestamp") for e in events if e.get("ts") or e.get("timestamp")]

        result = {
            "total_events": len(events),
            "unique_sources": len(source_counts),
            "top_sources": [{"source": s, "count": c} for s, c in top_sources],
            "buffer_capacity": _EVENT_STREAM.maxlen if hasattr(_EVENT_STREAM, "maxlen") else "unknown",
        }
        if timestamps:
            result["oldest_event_ts"] = str(min(timestamps))
            result["newest_event_ts"] = str(max(timestamps))
        return json.dumps(result, default=str)

    elif name == "clear_buffer":
        count_before = len(_EVENT_STREAM)
        _EVENT_STREAM.clear()
        return json.dumps({
            "status": "success",
            "events_cleared": count_before,
            "message": "StreamSync buffer successfully cleared."
        })

    return f"Error: Tool '{name}' not found in StreamSync context."
