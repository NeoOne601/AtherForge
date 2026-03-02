# AetherForge v1.0 — src/modules/localbuddy/graph.py
# ─────────────────────────────────────────────────────────────────
# LocalBuddy: Conversational AI with persistent multi-session memory.
# Handles general Q&A, coding, writing, and knowledge retrieval.
# Session memory is bounded to prevent unbounded RAM growth.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger("aetherforge.localbuddy")

# Session memory: session_id → deque of (role, content) pairs
# Max 50 turns per session (~100k tokens for context management)
_MEMORY: dict[str, deque[dict[str, str]]] = {}
_MAX_TURNS = 50


def build_localbuddy_graph() -> dict[str, Any]:
    """Build LocalBuddy module descriptor."""
    return {
        "module_id": "localbuddy",
        "run": run_localbuddy,
        "clear_memory": clear_session_memory,
        "get_history": get_session_history,
    }


def run_localbuddy(
    session_id: str,
    message: str,
    llm_fn: Any,  # Callable[[list[dict]], str] from meta_agent
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run LocalBuddy for one turn.
    Maintains rolling session history and passes full context to LLM.

    Args:
      session_id: Unique session identifier
      message:    User message text
      llm_fn:     Callable that takes a list of messages and returns str

    Returns dict with "response", "turn_count", "memory_used_pct"
    """
    # Get or create session memory
    if session_id not in _MEMORY:
        _MEMORY[session_id] = deque(maxlen=_MAX_TURNS)

    memory = _MEMORY[session_id]
    memory.append({"role": "user", "content": message})

    # Build messages list for LLM
    messages = [{"role": m["role"], "content": m["content"]} for m in memory]

    # Get LLM response
    try:
        response = llm_fn(messages) if callable(llm_fn) else f"Echo: {message}"
    except Exception as exc:
        logger.exception("LocalBuddy LLM call failed: %s", exc)
        response = "I encountered an error processing your request."

    memory.append({"role": "assistant", "content": response})

    return {
        "response": response,
        "turn_count": len(memory) // 2,
        "memory_used_pct": round(len(memory) / _MAX_TURNS * 100, 1),
    }


def clear_session_memory(session_id: str) -> bool:
    """Clear memory for a session. Returns True if session existed."""
    if session_id in _MEMORY:
        del _MEMORY[session_id]
        logger.info("Cleared LocalBuddy memory for session: %s", session_id)
        return True
    return False


def get_session_history(session_id: str) -> list[dict[str, str]]:
    """Return the full conversation history for a session."""
    return list(_MEMORY.get(session_id, []))
