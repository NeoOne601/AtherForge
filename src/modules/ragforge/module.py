from __future__ import annotations

from typing import Any

from src.modules.base import BaseModule
from src.modules.ragforge.cognitive_rag import CognitiveRAG


class RagForgeModule(BaseModule):
    """
    RagForge module for document intelligence and retrieval-augmented generation.
    """

    def __init__(self, cognitive_rag: CognitiveRAG):
        super().__init__(name="ragforge")
        self.cognitive_rag = cognitive_rag

    @property
    def system_prompt_extension(self) -> str:
        return (
            "\n\nYou are in RAGForge mode. Use the provided tools to search through "
            "uploaded documents. Always prefer grounded evidence over general knowledge."
        )

    async def initialize(self) -> None:
        """Module-specific initialization if needed."""
        pass

    async def process(self, payload: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """
        Execute the RAGForge thinking retrieval pipeline.
        Payload expected keys: 'message', 'context' (optional source_filter)
        """
        message = payload.get("message", "")
        context = payload.get("context", {})
        active_docs = context.get("active_docs", [])
        source_filter = active_docs[0] if len(active_docs) == 1 else active_docs

        # CognitiveRAG is typically synchronous in its current form,
        # but the interface allows for future async search/inference.
        answer, docs, trace = self.cognitive_rag.think_and_answer(
            query=message, source_filter=source_filter
        )

        return {
            "content": answer,
            "metadata": {
                "retrieved_docs": [d.metadata for d in docs],
                "thinking_trace": trace.__dict__ if hasattr(trace, "__dict__") else str(trace),
            },
            "causal_edges": [
                {"source": "rag_start", "target": "retrieval", "label": "Hybrid Search"},
                {"source": "retrieval", "target": "synthesis", "label": "CoT Synthesis"},
            ],
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return []

    def execute_tool(self, name: str, args: dict[str, Any], state: Any | None = None) -> Any:
        return f"Error: Tool '{name}' not found in RagForge module."
