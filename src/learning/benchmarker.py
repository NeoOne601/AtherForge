# AetherForge v1.0 — src/learning/benchmarker.py
# ─────────────────────────────────────────────────────────────────
# BitNet Training Benchmarker.
#
# Runs a fixed 2-minute training sprint to measure BPB improvement
# and convergence stability under current genome configurations.
# ─────────────────────────────────────────────────────────────────


import structlog

from src.learning.bitnet_trainer import BitNetTrainer

logger = structlog.get_logger("aetherforge.bitnet_benchmarker")


class BitNetBenchmarker:
    """
    Standardized benchmark for fine-tuning performance.
    """

    def __init__(self, trainer: BitNetTrainer) -> None:
        self.trainer = trainer

    async def run_sprint(self) -> float:
        """
        Runs a short OPLoRA training cycle and returns the BPB (Bits-Per-Byte).
        Lower BPB is better, so this returns (1.0 - normalized_bpb) for hill-climbing.
        """
        logger.info("Starting BitNet Training Sprint (Benchmark)...")
        try:
            # We run a standard cycle. In a real benchmark, we'd use a fixed test set.
            result = await self.trainer.run_oploora_cycle()

            # Metric for hill-climbing: Higher is better.
            # If BPB is typically 0.07, (1.0 - 0.07) = 0.93.
            metric = 1.0 - result.bpb
            logger.info("BitNet Sprint complete. BPB: %.4f | Fit Metric: %.4f", result.bpb, metric)
            return metric
        except Exception as e:
            logger.error("BitNet Benchmark sprint failed: %s", e)
            return 0.0
