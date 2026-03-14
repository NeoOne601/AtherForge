from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.tool_registry import tool_registry


class BaseModule(ABC):
    """
    Abstract base class for all AetherForge modules.
    Standardizes module lifecycle and tool registration (ISP, LSP).
    """

    def __init__(self, name: str):
        self.name = name

    async def initialize(self) -> None:
        """One-time initialization logic (e.g. loading heavy models)."""
        pass

    def validate_payload(self, payload: dict[str, Any]) -> bool:
        """Basic validation of incoming processing payloads."""
        return True

    @abstractmethod
    def system_prompt_extension(self) -> str:
        """Specific instructions this module adds to the global system prompt."""
        pass

    @abstractmethod
    async def process(self, payload: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """
        Execute the primary logic for this module.
        Returns a dict with 'content', 'metadata', and 'causal_edges'.
        """
        pass

    @abstractmethod
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return LLM tool definitions for this module."""
        return []

    def register_tools(self) -> None:
        """Register all module tools with the central ToolRegistry (OCP)."""
        definitions = self.get_tool_definitions()
        for df in definitions:
            name = df.get("name")
            if name:

                def _create_handler(n: str):
                    def _handler(args: dict[str, Any], state: Any | None = None) -> Any:
                        return self.execute_tool(n, args, state)

                    return _handler

                tool_registry.register_tool(df, _create_handler(name))

    @abstractmethod
    def execute_tool(self, name: str, args: dict[str, Any], state: Any | None = None) -> Any:
        """Execute a tool belonging to this module."""
        pass

    async def shutdown(self) -> None:
        """Optional shutdown logic for the module."""
        pass
