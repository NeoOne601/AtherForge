from __future__ import annotations
from typing import Any, Dict, List, Optional
import json

from src.modules.base import BaseModule
from src.modules.watchtower.tools import get_tools, execute_tool

class WatchTowerModule(BaseModule):
    """
    WatchTower module for system metrics and process management.
    """

    def __init__(self):
        super().__init__(name="watchtower")

    async def initialize(self) -> None:
        # No specific async init needed for WatchTower yet
        pass

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return get_tools()

    def execute_tool(self, name: str, args: Dict[str, Any], state: Optional[Any] = None) -> str:
        return execute_tool(name, args)
