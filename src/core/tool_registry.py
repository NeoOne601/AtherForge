from __future__ import annotations
import structlog
from typing import Any, Callable, Dict, List, Optional

logger = structlog.get_logger("aetherforge.core.tool_registry")

class ToolRegistry:
    """
    Central registry for LLM-callable tools.
    Follows the Open/Closed Principle (OCP) by allowing modules to 
    register tools without modifying the core agent logic.
    """
    _instance: Optional[ToolRegistry] = None

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._tools: dict[str, Any] = {}
            self._handlers: dict[str, Callable[..., Any]] = {}
            self._initialized = True

    def register_tool(self, definition: dict[str, Any], handler: Callable[..., Any]) -> None:
        """Register a new tool with its definition and execution handler."""
        name = definition.get("name")
        if not name:
            raise ValueError("Tool definition must include a 'name' field.")
        
        self._tools[name] = definition
        self._handlers[name] = handler
        logger.debug("Tool registered", name=name)

    def get_tool_definitions(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return a list of tool definitions, optionally filtered by name."""
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def execute_tool(self, name: str, args: Dict[str, Any], state: Optional[Any] = None) -> Any:
        """Execute a registered tool by name with provided arguments."""
        if name not in self._handlers:
            logger.error("Attempted to execute unregistered tool", name=name)
            raise ValueError(f"Tool '{name}' is not registered.")
        
        handler = self._handlers[name]
        try:
            # Check if handler accepts state
            import inspect
            sig = inspect.signature(handler)
            if "state" in sig.parameters:
                return handler(args, state=state)
            return handler(args)
        except Exception as e:
            logger.exception("Error executing tool", name=name, error=str(e))
            raise

tool_registry = ToolRegistry()
