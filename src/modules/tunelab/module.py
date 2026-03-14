# AetherForge v1.0 — src/modules/tunelab/module.py
from __future__ import annotations

from typing import Any

from src.modules.base import BaseModule
from src.modules.tunelab.tools import execute_tool as exec_tool
from src.modules.tunelab.tools import get_tools


class TuneLabModule(BaseModule):
    """
    Standardized TuneLab module for offline learning and OPLoRA tuning.
    """

    def __init__(self, settings: Any, replay_buffer: Any):
        super().__init__(name="tunelab")
        self.settings = settings
        self.replay_buffer = replay_buffer

    @property
    def system_prompt_extension(self) -> str:
        return (
            "You are the TuneLab AI. Your job is to EXECUTE training operations and REPORT LIVE DATA — not explain theory. "
            "RULES: (1) When the user asks how many samples are ready, pending, or in queue, call query_buffer_stats immediately. "
            "(2) When the user says compile, train, trigger, or start — call trigger_compilation immediately. "
            "(3) Never say you don't have access to data. Use your tools to fetch it."
        )

    async def process(self, payload: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """Placeholder for custom TuneLab processing logic."""
        return {
            "content": "TuneLab module ready for training orchestration.",
            "metadata": {"status": "active"},
            "causal_edges": [],
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return get_tools()

    def execute_tool(self, name: str, args: dict[str, Any], state: Any | None = None) -> Any:
        # We pass self as a pseudo-state if needed, but the original exec_tool expects an object with settings/replay_buffer
        class MockState:
            def __init__(self, s, r):
                self.settings = s
                self.replay_buffer = r

        return exec_tool(name, args, MockState(self.settings, self.replay_buffer))
