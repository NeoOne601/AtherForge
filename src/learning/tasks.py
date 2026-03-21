# AetherForge v1.0 — src/learning/tasks.py
import structlog
from fastapi import FastAPI
from src.schemas import AppState

logger = structlog.get_logger("aetherforge.learning.tasks")

async def _nightly_oplora_job(app: FastAPI, force: bool = False) -> None:
    """Nightly background job for OPLoRA fine-tuning."""
    state: AppState = app.state.app_state
    
    if state.oplora_running:
        logger.warning("OPLoRA job already in progress; skipping trigger.")
        return

    state.oplora_running = True
    logger.info("OPLoRA job starting", force=force)
    try:
        from src.learning.bitnet_trainer import BitNetTrainer
        trainer = BitNetTrainer(state.settings, state.replay_buffer)
        await trainer.run_oploora_cycle(force=force)
        logger.info("OPLoRA job complete")
    except Exception as e:
        logger.error("OPLoRA job critical failure", error=str(e))
    finally:
        state.oplora_running = False
