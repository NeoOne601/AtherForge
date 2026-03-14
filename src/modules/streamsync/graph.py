# AetherForge v1.0 — src/modules/streamsync/graph.py
# ─────────────────────────────────────────────────────────────────
# StreamSync: Event stream processing and temporal pattern recognition.
# Processes event sequences, finds correlations, and surfaces insights.
# Uses a sliding-window pattern matcher — no external stream processors.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger("aetherforge.streamsync")

# Global event stream (bounded ring buffer)
_EVENT_STREAM: deque[dict[str, Any]] = deque(maxlen=10_000)
_PATTERN_REGISTRY: dict[str, list[str]] = {}  # pattern_name → [event sequence]


@dataclass
class StreamEvent:
    """A single event in the stream."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)


def build_streamsync_graph() -> dict[str, Any]:
    """Build StreamSync module descriptor."""
    return {
        "module_id": "streamsync",
        "run": run_streamsync,
        "emit": emit_event,
        "register_pattern": register_pattern,
        "get_stats": get_stream_stats,
    }


def emit_event(event_type: str, payload: dict[str, Any] | None = None, source: str = "api") -> str:
    """Emit a single event into the stream. Returns event ID."""
    import uuid

    event_id = str(uuid.uuid4())
    event: dict[str, Any] = {
        "id": event_id,
        "event_type": event_type,
        "payload": payload or {},
        "source": source,
        "timestamp": time.time(),
    }
    _EVENT_STREAM.append(event)
    logger.debug("Event emitted: %s from %s", event_type, source)
    return event_id


def register_pattern(name: str, sequence: list[str]) -> None:
    """Register a named event sequence pattern for detection."""
    _PATTERN_REGISTRY[name] = sequence
    logger.info("Registered pattern '%s': %s", name, sequence)


def _detect_patterns(window: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sliding window pattern detection. O(N × P) where P = pattern count.
    Returns list of detected pattern matches with timestamps.
    """
    event_types = [e["event_type"] for e in window]
    matches = []

    for pattern_name, sequence in _PATTERN_REGISTRY.items():
        k = len(sequence)
        for i in range(len(event_types) - k + 1):
            if event_types[i : i + k] == sequence:
                matches.append(
                    {
                        "pattern": pattern_name,
                        "start_idx": i,
                        "end_idx": i + k - 1,
                        "start_ts": window[i]["timestamp"],
                        "end_ts": window[i + k - 1]["timestamp"],
                    }
                )
    return matches


def run_streamsync(
    query: str,
    window_size: int = 100,
    event_types_filter: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Analyze the event stream and return insights.

    Args:
      query:               Natural language analysis request
      window_size:         How many recent events to analyze
      event_types_filter:  Restrict analysis to these event types

    Returns structured stream analysis with patterns and statistics.
    """
    # Snapshot recent events
    recent = list(_EVENT_STREAM)[-window_size:]
    if event_types_filter:
        recent = [e for e in recent if e["event_type"] in event_types_filter]

    if not recent:
        return {
            "query": query,
            "events_analyzed": 0,
            "patterns": [],
            "top_event_types": [],
            "summary": "No events in stream",
        }

    # ── Frequency analysis ────────────────────────────────────────
    type_counts = Counter(e["event_type"] for e in recent)
    top_types = [{"type": t, "count": c} for t, c in type_counts.most_common(10)]

    # ── Source analysis ───────────────────────────────────────────
    source_counts = Counter(e["source"] for e in recent)

    # ── Pattern detection ─────────────────────────────────────────
    patterns = _detect_patterns(recent) if _PATTERN_REGISTRY else []

    # ── Temporal analysis ─────────────────────────────────────────
    if len(recent) >= 2:
        duration = recent[-1]["timestamp"] - recent[0]["timestamp"]
        events_per_sec = len(recent) / max(duration, 0.001)
    else:
        events_per_sec = 0.0

    summary = (
        f"Analyzed {len(recent)} events from {len(source_counts)} sources. "
        f"Rate: {events_per_sec:.1f} events/sec. "
        f"Top type: {top_types[0]['type'] if top_types else 'none'}. "
        f"Patterns detected: {len(patterns)}."
    )

    logger.info("StreamSync analysis: %s", summary)
    return {
        "query": query,
        "events_analyzed": len(recent),
        "top_event_types": top_types,
        "source_distribution": dict(source_counts),
        "patterns": patterns,
        "events_per_sec": round(events_per_sec, 2),
        "summary": summary,
    }


def get_stream_stats() -> dict[str, Any]:
    """Return global stream statistics."""
    return {
        "total_events_buffered": len(_EVENT_STREAM),
        "registered_patterns": len(_PATTERN_REGISTRY),
        "buffer_capacity": _EVENT_STREAM.maxlen,
    }
