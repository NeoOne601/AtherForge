# AetherForge v1.0 — src/modules/sync/module.py
from __future__ import annotations

from typing import Any

from src.modules.base import BaseModule


class SyncModule(BaseModule):
    """
    Standardized Sync module for P2P Zero-Knowledge synchronization.
    """

    def __init__(self, sync_manager: Any):
        super().__init__(name="sync")
        self.sync_manager = sync_manager

    @property
    def system_prompt_extension(self) -> str:
        return (
            "You are the Sync AI. Your job is to EXECUTE sync operations — not explain it. "
            "RULES: (1) When the user asks to see, show, or analyze events, call query_stream immediately. "
            "(2) When the user asks for an overview, summary, or patterns, call summarize_stream immediately. "
            "(3) When the user asks to clear or reset the buffer, call clear_buffer immediately. "
            "Never explain how sync works when you can just show real data instead."
        )

    async def process(self, payload: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """Placeholder for custom Sync processing logic."""
        return {
            "content": "Sync module ready for P2P synchronization.",
            "metadata": {"status": "active"},
            "causal_edges": [],
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        # SyncModule tools are currently handled via StreamSync context in MetaAgent
        # We'll keep it empty for now or add basic discovery tools
        return []

    def execute_tool(self, name: str, args: dict[str, Any], state: Any | None = None) -> Any:
        return f"Error: Tool '{name}' not found in Sync context."
