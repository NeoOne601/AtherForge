from __future__ import annotations

from typing import Any

from src.modules.base import BaseModule
from src.modules.watchtower.tools import execute_tool, get_tools


class WatchTowerModule(BaseModule):
    """
    WatchTower module for system metrics and process management.
    """

    def __init__(self):
        super().__init__(name="watchtower")

    @property
    def system_prompt_extension(self) -> str:
        return (
            "\n\nYou are in WatchTower mode. Use the provided tools to monitor system "
            "metrics, processes, and resource utilization. Present technical data "
            "clearly and alert the user to any anomalies."
        )

    async def initialize(self) -> None:
        # No specific async init needed for WatchTower yet
        pass

    async def process(self, payload: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """
        Execute WatchTower logic.
        For now, this is a pass-through to the tool execution loop.
        """
        return {
            "content": "[WatchTower] System monitoring protocol active. Awaiting metric query.",
            "metadata": {},
            "causal_edges": [
                {"source": "watchtower_start", "target": "ready", "label": "Monitoring Active"}
            ],
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return get_tools()

    def execute_tool(self, name: str, args: dict[str, Any], state: Any | None = None) -> str:
        return execute_tool(name, args)
