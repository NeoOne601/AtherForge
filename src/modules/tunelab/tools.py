# AetherForge v1.0 — src/modules/tunelab/tools.py
import json
import structlog
import threading
import asyncio
from typing import Any

from src.learning.replay_buffer import ReplayBuffer

logger = structlog.get_logger("aetherforge.tunelab.tools")

def get_tools() -> list[dict[str, Any]]:
    """Return TuneLab-specific LLM tool definitions."""
    return [
        {
            "name": "query_buffer_stats",
            "description": (
                "Get live statistics about the OPLoRA Replay Buffer — "
                "how many samples are ready for training, filtered out, already compiled, and pending. "
                "CALL THIS IMMEDIATELY when the user asks: how many are ready, how full is the buffer, "
                "what is the training queue size, or any question about sample counts or training status."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "trigger_compilation",
            "description": (
                "Trigger an immediate OPLoRA compilation cycle to learn from the Replay Buffer. "
                "CALL THIS when the user says: compile, train, trigger, start, run, or begin the OPLoRA/training cycle."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]

def execute_tool(name: str, args: dict[str, Any], state: Any) -> str:
    """Execute a TuneLab tool and return string result."""
    logger.info("TuneLab Tool Execution: %s(%s)", name, args)

    if name == "trigger_compilation":
        from src.learning.bitnet_trainer import BitNetTrainer
        trainer = BitNetTrainer(state.settings, state.replay_buffer)

        def _run_trigger() -> None:
            asyncio.run(trainer.run_oploora_cycle())

        threading.Thread(target=_run_trigger, daemon=True).start()
        return json.dumps({
            "status": "started",
            "message": "OPLoRA compilation pipeline triggered successfully. The training cycle is running in the background."
        })

    elif name == "query_buffer_stats":
        replay_buffer: ReplayBuffer = state.replay_buffer
        if replay_buffer is None:
            return json.dumps({"error": "Replay Buffer not initialized."})

        try:
            # Run the async stats call synchronously
            loop = asyncio.new_event_loop()
            stats = loop.run_until_complete(replay_buffer.get_stats())
            loop.close()

            # Enrich with readiness counts if available
            total = stats.get("total_items", 0)
            ready = stats.get("ready_for_training", stats.get("high_fidelity_count", total))
            filtered = stats.get("filtered_count", stats.get("low_fidelity_count", 0))
            compiled = stats.get("compiled_count", 0)

            result = {
                "total_samples": total,
                "ready_for_training": ready,
                "filtered_noise": filtered,
                "already_compiled": compiled,
                "buffer_capacity": stats.get("capacity", "unknown"),
                "min_faithfulness_threshold": stats.get("min_faithfulness", 0.92),
            }
            return json.dumps(result)
        except Exception as e:
            logger.error("query_buffer_stats failed: %s", e)
            return json.dumps({"error": str(e), "total_samples": 0, "ready_for_training": 0})

    return f"Error: Tool '{name}' not found in TuneLab context."
