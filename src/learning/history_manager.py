# AetherForge v1.0 — src/learning/history_manager.py
# ─────────────────────────────────────────────────────────────────
# Training history persistence for TuneLab.
# Stores TrainingResult objects as a time-series JSON log.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import time
from typing import Any

import structlog

from src.config import AetherForgeSettings

logger = structlog.get_logger("aetherforge.history_manager")


class HistoryManager:
    """
    Manages persistence of training metrics history.
    Backed by a simple append-only JSON file.
    """

    def __init__(self, settings: AetherForgeSettings) -> None:
        self.settings = settings
        self._history_file = settings.data_dir / "training_history.json"
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # [RUVECTOR PHASE 6 PLACEHOLDER] SONA LoRA Updater
        # Replaces OPLoRA with Sparse Orthogonal N-dimensional Adaptation
        try:
            from ruvector_sona import SonaUpdater
            self.sona_updater = SonaUpdater(str(settings.data_dir / "sona_weights.bin"))
        except ImportError:
            self.sona_updater = None

    def record(self, result: Any) -> None:
        """
        Record a training result to history.
        `result` is expected to be a TrainingResult-compatible dict or object.
        """
        try:
            # Convert TrainingResult dataclass to dict if needed
            if hasattr(result, "__dict__"):
                entry = result.__dict__.copy()
            else:
                entry = dict(result)

            entry["timestamp"] = time.time()

            history = self.get_history()
            history.append(entry)

            # Keep only last 100 runs to prevent file bloat
            if len(history) > 100:
                history = history[-100:]

            with open(self._history_file, "w") as f:
                json.dump(history, f, indent=2)

            logger.info(
                "Recorded training result to history: loss=%.4f", entry.get("avg_loss", 0.0)
            )
        except Exception as e:
            logger.error("Failed to record training history: %s", e)

    def get_history(self) -> list[dict[str, Any]]:
        """Return the training history list."""
        if not self._history_file.exists():
            return []
        try:
            with open(self._history_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read training history: %s", e)
            return []
