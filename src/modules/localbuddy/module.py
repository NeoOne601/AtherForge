# AetherForge v1.0 — src/modules/localbuddy/module.py
from __future__ import annotations

from typing import Any

from src.modules.base import BaseModule


class LocalBuddyModule(BaseModule):
    """
    Standardized LocalBuddy module for general conversational AI.
    """

    def __init__(self):
        super().__init__(name="localbuddy")

    @property
    def system_prompt_extension(self) -> str:
        return (
            "You are in LocalBuddy mode. Act as a deeply helpful, technically precise AI assistant. "
            "For any non-trivial question, follow this pattern:\n"
            "1) Briefly restate the user's goal in your own words.\n"
            "2) Outline a short plan or set of steps you will take.\n"
            "3) Work through the steps one by one, explaining key reasoning.\n"
            "4) Finish with a concise summary or recommended next actions.\n"
            "Always remember conversation context and be explicit about assumptions or uncertainties."
        )

    async def process(self, payload: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """Placeholder for custom LocalBuddy processing logic."""
        return {
            "content": "LocalBuddy ready for conversational assistance.",
            "metadata": {"status": "active"},
            "causal_edges": [],
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        # LocalBuddy currently has no specific tools of its own
        return []

    def execute_tool(self, name: str, args: dict[str, Any], state: Any | None = None) -> Any:
        return f"Error: Tool '{name}' not found in LocalBuddy context."
