# AetherForge v1.0 — src/learning/evolution.py
# ─────────────────────────────────────────────────────────────────
# Autonomous Hill-Climbing for AetherForge System Parameters.
#
# Inspired by karpathy/autoresearch, this module enables AetherForge
# to evolve its own "Genome" (configuration) by running controlled
# experiments and measure performance against a fixed metric (BPB).
# ─────────────────────────────────────────────────────────────────

import json
import random
import time
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.config import AetherForgeSettings

logger = structlog.get_logger("aetherforge.evolution")


class SelfOptGenome(BaseModel):
    """The 'tunable genes' of AetherForge."""

    # RAG Genes
    rag_chunk_size: int = Field(default=512, ge=128, le=2048)
    rag_overlap: int = Field(default=50, ge=0, le=200)
    rag_top_k: int = Field(default=5, ge=1, le=20)

    # Learning Genes
    learning_rate: float = Field(default=3e-4, ge=1e-6, le=1e-2)
    lora_rank: int = Field(default=16, ge=4, le=64)
    lora_alpha: int = Field(default=32, ge=8, le=128)

    # System Genes
    deep_reasoning_temp: float = Field(default=0.7, ge=0.0, le=1.5)
    faithfulness_threshold: float = Field(default=0.85, ge=0.5, le=1.0)

    # Prompt Genes
    system_prompt_variant: str = Field(default="v1")
    rag_prompt_variant: str = Field(default="v1")


class ExperimentRecord(BaseModel):
    """Metadata for a single mutation experiment."""

    experiment_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    mutation_target: str
    initial_value: Any
    new_value: Any
    baseline_metric: float  # BPB or Grounding Score
    new_metric: float
    status: str = "pending"  # kept, rolled_back, failed


class ExperimentManager:
    """Orchestrates the evolution loop."""

    def __init__(self, settings: AetherForgeSettings) -> None:
        self.settings = settings
        self.genome_path = settings.data_dir / "evolution_genome.json"
        self.history_path = settings.data_dir / "evolution_history.jsonl"
        self._load_genome()

    def _load_genome(self) -> None:
        if self.genome_path.exists():
            try:
                data = json.loads(self.genome_path.read_text())
                self.current_genome = SelfOptGenome.model_validate(data)
            except Exception as e:
                logger.error("Failed to load genome, resetting to defaults: %s", e)
                self.current_genome = SelfOptGenome()
        else:
            self.current_genome = SelfOptGenome()
            self._save_genome()

    def _save_genome(self) -> None:
        self.genome_path.write_text(self.current_genome.model_dump_json(indent=2))

    def propose_mutation(self) -> dict[str, Any]:
        """Proposes a single mutation to the current genome."""
        genes = list(self.current_genome.model_fields.keys())
        target_gene = random.choice(genes)

        current_val = getattr(self.current_genome, target_gene)

        # Simple mutation logic: +/- 10% or +/- fixed increment
        if isinstance(current_val, int):
            delta = random.choice([-1, 1]) * max(1, int(current_val * 0.1))
            new_val = current_val + delta
        else:
            delta = random.choice([-1, 1]) * (current_val * 0.1)
            new_val = current_val + delta

        # Clamp values based on Pydantic field constraints (simplified for now)
        # In a real impl, we'd use the le/ge constraints from the model.

        return {"gene": target_gene, "old": current_val, "new": new_val}

    def apply_mutation(self, gene: str, value: Any) -> None:
        """Applies a mutation to the live genome."""
        setattr(self.current_genome, gene, value)
        self._save_genome()
        logger.info("Genome mutated: %s -> %s", gene, value)

    def record_result(self, record: ExperimentRecord) -> None:
        """Persists experiment result to history."""
        with open(self.history_path, "a") as f:
            f.write(record.model_dump_json() + "\n")
        logger.info(
            "Experiment recorded: %s (Metric: %.4f -> %.4f)",
            record.experiment_id,
            record.baseline_metric,
            record.new_metric,
        )

    async def run_experiment(self, mutation: dict[str, Any], benchmark_fn: Any) -> ExperimentRecord:
        """
        Executes a full hill-climbing experiment cycle:
        1. Measure Baseline
        2. Apply Mutation
        3. Measure New Performance
        4. Decide (Keep/Rollback)
        """
        exp_id = f"exp_{int(time.time())}"
        baseline_metric = await benchmark_fn()

        # Apply
        self.apply_mutation(mutation["gene"], mutation["new"])

        # Wait for system to stabilize or run fixed training cycle (demo uses sleep)
        # In production, this would trigger bitnet_trainer.run_oploora_cycle()
        new_metric = await benchmark_fn()

        status = "kept"
        # Hill-climbing logic: higher is better (metric is 1.0 - BPB or grounding)
        if new_metric < baseline_metric:
            logger.info(
                "Mutation worsened performance (%.4f < %.4f). Rolling back...",
                new_metric,
                baseline_metric,
            )
            self._git_rollback()
            self.apply_mutation(mutation["gene"], mutation["old"])
            status = "rolled_back"
        else:
            logger.info(
                "Mutation improved performance (%.4f >= %.4f). Keeping change.",
                new_metric,
                baseline_metric,
            )

        record = ExperimentRecord(
            experiment_id=exp_id,
            mutation_target=mutation["gene"],
            initial_value=mutation["old"],
            new_value=mutation["new"],
            baseline_metric=baseline_metric,
            new_metric=new_metric,
            status=status,
        )
        self.record_result(record)
        return record

    def _git_rollback(self) -> None:
        """Uses git to revert any changes to the genome file if mutation fails."""
        import subprocess

        try:
            # Only rollback the genome file to keep other system logs intact
            subprocess.run(["git", "checkout", str(self.genome_path)], check=False)
        except Exception as e:
            logger.error("Git rollback failed: %s", e)


class AetherResearcher:
    """The agent specialization that 'dreams' of improvements."""

    def __init__(self, manager: ExperimentManager) -> None:
        self.manager = manager

    async def run_evolution_cycle(self, benchmark_fn: Any) -> ExperimentRecord:
        """Runs a single autonomous research cycle using the provided benchmarker."""
        mutation = self.manager.propose_mutation()
        return await self.manager.run_experiment(mutation, benchmark_fn)
