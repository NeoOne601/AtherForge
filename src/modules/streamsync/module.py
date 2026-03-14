from __future__ import annotations

from typing import Any

from src.modules.base import BaseModule
from src.modules.streamsync.tools import execute_tool, get_tools


class StreamSyncModule(BaseModule):
    """
    StreamSync module for event streams and RSS feeds.
    """

    def __init__(self):
        super().__init__(name="streamsync")

    @property
    def system_prompt_extension(self) -> str:
        return (
            "\n\nYou are in StreamSync mode. Use the provided tools to manage event "
            "streams, RSS feeds, and directory monitoring. Help the user stay updated "
            "with real-time information flows."
        )

    async def initialize(self) -> None:
        # No specific async init needed yet
        pass

    async def process(self, payload: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """
        Execute StreamSync logic.
        """
        return {
            "content": "[StreamSyncModule] Stream engine active. Monitoring feeds.",
            "metadata": {},
            "causal_edges": [
                {"source": "streamsync_start", "target": "ready", "label": "Engine Active"}
            ],
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return get_tools()

    def execute_tool(self, name: str, args: dict[str, Any], state: Any | None = None) -> str:
        return execute_tool(name, args)
