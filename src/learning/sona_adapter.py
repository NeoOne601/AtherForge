# AetherForge v1.0 — src/learning/sona_adapter.py
# ─────────────────────────────────────────────────────────────────
# SONA 3-tier learning adapter.
# Supplements OPLoRA nightly batch with per-request adaptation.
#
# Tier 1: MicroLoRA rank-2 — adapts in <1ms per accepted response
# Tier 2: EWC++ consolidation — runs in background ~100ms
# Tier 3: ReasoningBank — stores successful query→answer trajectories
#
# OPLoRA nightly runs continue unchanged. SONA supplements, not replaces.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SONAAdapter:
    """
    SONA 3-tier learning adapter.
    Supplements OPLoRA nightly batch with per-request adaptation.

    Lifecycle:
      1. initialize() — called at startup, fails gracefully if unavailable
      2. on_interaction() — called after every completed interaction
      3. get_stats() — returns learning stats for TuneLab display
    """

    def __init__(self, model_path: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self._sona: Any = None
        self._model_path = model_path
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize SONA. Called at startup. Fails gracefully if unavailable."""
        try:
            from ruvector_sona import SONA  # type: ignore

            self._sona = SONA(
                model_path=self._model_path,
                micro_lora_rank=2,        # <1ms per-request adaptation
                ewc_lambda=400,           # catastrophic forgetting protection
                reasoning_bank=True,      # trajectory curriculum memory
                checkpoint_dir=str(self.data_dir / "sona_checkpoints"),
            )
            self._initialized = True
            logger.info("SONA adapter initialised")
        except ImportError:
            logger.warning(
                "ruvector-sona not installed — SONA learning disabled, "
                "will use OPLoRA nightly batch only"
            )
            self._initialized = False
        except Exception as e:
            logger.warning("SONA initialisation failed (will use OPLoRA only): %s", e)
            self._initialized = False

    async def on_interaction(
        self,
        query: str,
        response: str,
        verdict: str,            # "accepted" | "rejected" | "corrected"
        route: str = "synthesis",
        metadata: dict | None = None,
    ) -> None:
        """
        Called after every completed interaction.
        verdict = "accepted" when user accepts the response without correction.
        verdict = "rejected" when user corrects or dismisses the response.
        """
        if not self._initialized or self._sona is None:
            return
        try:
            # Tier 1: MicroLoRA instant adaptation
            await self._sona.micro_update(query, response, verdict)
            # Tier 3: Record to ReasoningBank for curriculum learning
            self._sona.reasoning_bank.record(
                query=query,
                response=response,
                verdict=verdict,
                metadata={"route": route, **(metadata or {})},
            )
        except Exception as e:
            logger.warning("SONA on_interaction failed (non-fatal): %s", e)

    async def get_stats(self) -> dict:
        """Return SONA learning stats for TuneLab display."""
        if not self._initialized or self._sona is None:
            return {"status": "unavailable", "initialized": False}
        try:
            return {
                "status": "active",
                "initialized": True,
                "micro_lora_rank": self._sona.micro_lora_rank,
                "reasoning_bank_size": self._sona.reasoning_bank.size(),
            }
        except Exception as e:
            logger.warning("SONA get_stats failed: %s", e)
            return {"status": "error", "initialized": True, "error": str(e)}
