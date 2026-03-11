from __future__ import annotations
from typing import Any, Dict, List, Optional

from src.modules.base import BaseModule
from src.modules.streamsync.tools import get_tools, execute_tool

class StreamSyncModule(BaseModule):
    """
    StreamSync module for event streams and RSS feeds.
    """

    def __init__(self):
        super().__init__(name="streamsync")

    async def initialize(self) -> None:
        # No specific async init needed yet
        pass

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return get_tools()

    def execute_tool(self, name: str, args: Dict[str, Any], state: Optional[Any] = None) -> str:
        return execute_tool(name, args)
