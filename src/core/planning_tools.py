# AetherForge v1.0 — src/core/planning_tools.py
# ─────────────────────────────────────────────────────────────────
# Planning & Research Tools — Dynamic Task Orchestration.
# ─────────────────────────────────────────────────────────────────

import json
from typing import Any

from .tool_registry import tool_registry


def write_todos(args: dict[str, Any], state: Any = None) -> str: # noqa: ANN401
    """
    Update the agent's task list (To-Do List).
    Args:
        todos: List of task descriptions.
    """
    todos = args.get("todos", [])
    if not state or not hasattr(state, "todo_list"):
        return "Error: Planning state not available."

    state.todo_list = todos
    return f"To-Do list updated with {len(todos)} tasks."

def write_vfs_note(args: dict[str, Any], state: Any = None) -> str: # noqa: ANN401
    """
    Write an intermediate research finding to the Virtual File System (VFS).
    Args:
        title: Short title of the finding.
        content: Detailed explanation or data.
    """
    title = args.get("title")
    content = args.get("content")

    if not state or not hasattr(state, "vfs"):
        return "Error: VFS not available."

    state.vfs.write_note(title, content)
    return f"Note '{title}' saved to VFS scratchpad."

def clear_planner(args: dict[str, Any], state: Any = None) -> str: # noqa: ANN401
    """
    Clear the current planner (To-Do List and Scratchpad) to free up memory.
    """
    if not state:
        return "Error: State not available."

    if hasattr(state, "todo_list"):
        state.todo_list = []

    if hasattr(state, "vfs"):
        state.vfs.purge()

    return "Planner and VFS scratchpad have been cleared."

def get_research_status(args: dict[str, Any], state: Any = None) -> str: # noqa: ANN401
    """
    Retrieve the current status of the research, including to-dos and notes.
    """
    if not state:
        return "Error: State not available."

    status = {
        "todos": getattr(state, "todo_list", []),
        "scratchpad_summary": state.vfs.get_summary() if hasattr(state, "vfs") else "N/A"
    }
    return json.dumps(status, indent=2)

# Definitions for registry
WRITE_TODOS_DEF = {
    "name": "write_todos",
    "description": (
        "Create or update a list of sub-tasks for a complex query. "
        "Use this for multi-step research."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of strings representing the steps to be taken."
            }
        },
        "required": ["todos"]
    }
}

WRITE_VFS_NOTE_DEF = {
    "name": "write_vfs_note",
    "description": (
        "Save an intermediate finding, fact, or insight to the internal scratchpad. "
        "This helps offload context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Subject of the note."},
            "content": {"type": "string", "description": "The actual finding or data retrieved."}
        },
        "required": ["title", "content"]
    }
}

CLEAR_PLANNER_DEF = {
    "name": "clear_planner",
    "description": (
        "Discard all current to-dos and research notes. "
        "Use this when the task is complete to save memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {}
    }
}

GET_RESEARCH_STATUS_DEF = {
    "name": "get_research_status",
    "description": "View current to-dos and a summary of notes saved in the VFS.",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}

def register_planning_tools() -> None:
    """Entry point to register these tools in the global registry."""
    tool_registry.register_tool(WRITE_TODOS_DEF, write_todos)
    tool_registry.register_tool(WRITE_VFS_NOTE_DEF, write_vfs_note)
    tool_registry.register_tool(CLEAR_PLANNER_DEF, clear_planner)
    tool_registry.register_tool(GET_RESEARCH_STATUS_DEF, get_research_status)
