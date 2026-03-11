from __future__ import annotations
from typing import Any, Dict, List, Optional

from src.modules.base import BaseModule
from src.modules.ragforge.cognitive_rag import CognitiveRAG

class RagForgeModule(BaseModule):
    """
    RagForge module for document intelligence and retrieval-augmented generation.
    """

    def __init__(self, cognitive_rag: CognitiveRAG):
        super().__init__(name="ragforge")
        self.cognitive_rag = cognitive_rag

    async def initialize(self) -> None:
        pass

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        # RagForge doesn't expose tools to the LLM the same way others do;
        # it's usually handled via direct module transition.
        return []

    def execute_tool(self, name: str, args: Dict[str, Any], state: Optional[Any] = None) -> str:
        # If we had tools like "summarize_document", they'd go here.
        return f"Error: Tool '{name}' not found in RagForge module."
