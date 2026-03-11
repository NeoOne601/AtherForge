from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.tool_registry import tool_registry

class BaseModule(ABC):
    """
    Abstract base class for all AetherForge modules.
    Standardizes module lifecycle and tool registration (ISP, LSP).
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize module-specific resources."""
        pass

    @abstractmethod
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return LLM tool definitions for this module."""
        return []

    def register_tools(self) -> None:
        """Register all module tools with the central ToolRegistry (OCP)."""
        definitions = self.get_tool_definitions()
        for df in definitions:
            name = df.get("name")
            if name:
                def _create_handler(n: str):
                    def _handler(args: Dict[str, Any], state: Optional[Any] = None) -> str:
                        return self.execute_tool(n, args, state)
                    return _handler
                tool_registry.register_tool(df, _create_handler(name))

    @abstractmethod
    def execute_tool(self, name: str, args: Dict[str, Any], state: Optional[Any] = None) -> str:
        """Execute a tool belonging to this module."""
        pass

    async def shutdown(self) -> None:
        """Optional shutdown logic for the module."""
        pass
