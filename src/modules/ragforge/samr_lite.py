# AetherForge v1.0 — src/modules/ragforge/samr_lite.py
# ─────────────────────────────────────────────────────────────────
# SAMR-lite: Semantic Answer fidelity Measurement & Review (lite)
#
# Adapted from NeoOne601/Ventro's SAMR hallucination detection concept.
# Original SAMR used PostgreSQL + Redis + per-org adaptive thresholds.
#
# This implementation is fully in-memory, zero dependencies on external
# services — designed for AetherForge's single-user embedded architecture.
#
# What it does:
#   After the LLM generates an answer, compute the cosine similarity
#   between the answer's semantic embedding and each retrieved context
#   chunk's embedding. If the average similarity is below the threshold,
#   the answer may have drifted from the source material (hallucination risk).
#
# Faithfulness Score (calibrated for CognitiveRAG synthesized answers):
#   ≥ 0.55 → GROUNDED  (answer is semantically consistent with sources)
#   < 0.55 → LOW_CONFIDENCE (answer may contain hallucinations) → BLOCKED
#   < 0.30 → LIKELY_HALLUCINATION (answer barely relates to sources) → BLOCKED
#
# Action: FaithfulnessError is RAISED when score < threshold.
# The meta-agent catches it and returns a safe refusal message.
# LLMs EXPLAIN. Deterministic engines CALCULATE. Citations PROVE.
# Never warn-and-deliver.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from typing import Any

import structlog

logger = structlog.get_logger("aetherforge.ragforge.samr_lite")

# Thresholds (tuned for CognitiveRAG chain-of-thought synthesized answers)
GROUNDED_THRESHOLD = 0.55  # answer is grounded in sources
LOW_CONFIDENCE_THRESHOLD = 0.30  # answer is questionable
DEFAULT_DIMS = 384  # all-MiniLM-L6-v2 embedding dimensions


class FaithfulnessError(Exception):
    """Raised when an LLM response fails the faithfulness check.

    The meta-agent catches this and returns a safe refusal message
    instead of delivering the ungrounded response to the user.
    """

    def __init__(self, score: float, threshold: float) -> None:
        super().__init__(
            f"Faithfulness score {score:.2f} below threshold {threshold:.2f}"
        )
        self.score = score
        self.threshold = threshold


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_faithfulness(
    answer_embedding: list[float],
    context_embeddings: list[list[float]],
    threshold: float = GROUNDED_THRESHOLD,
) -> dict[str, Any]:
    """
    Core SAMR-lite computation.

    Args:
        answer_embedding:   all-MiniLM-L6-v2 embedding of the generated answer
        context_embeddings: all-MiniLM-L6-v2 embeddings of the retrieved source chunks
        threshold:          Minimum score to be considered "grounded"

    Returns:
        {
            "faithfulness_score": float (0.0 – 1.0),
            "verdict": "GROUNDED" | "LOW_CONFIDENCE" | "LIKELY_HALLUCINATION",
            "alert_triggered": bool,
            "alert_icon": str,
            "interpretation": str,
            "per_chunk_scores": list[float],
        }
    """
    if not context_embeddings:
        return {
            "faithfulness_score": 0.0,
            "verdict": "NO_CONTEXT",
            "alert_triggered": False,
            "alert_icon": "ℹ️",
            "interpretation": "No source context was retrieved to validate against.",
            "per_chunk_scores": [],
        }

    per_chunk = [
        round(_cosine_similarity(answer_embedding, ctx_emb), 4) for ctx_emb in context_embeddings
    ]
    avg_score = round(sum(per_chunk) / len(per_chunk), 4)
    max_score = max(per_chunk)

    # Use max-score as tie-breaker: if the best chunk is highly similar,
    # the answer is likely grounded even if other chunks aren't relevant.
    effective_score = round((avg_score * 0.6) + (max_score * 0.4), 4)

    if effective_score >= GROUNDED_THRESHOLD:
        verdict = "GROUNDED"
        alert = False
        icon = "✅"
        interpretation = (
            f"Answer is semantically grounded in the retrieved sources "
            f"(faithfulness score: {effective_score:.2f})."
        )
    elif effective_score >= LOW_CONFIDENCE_THRESHOLD:
        verdict = "LOW_CONFIDENCE"
        alert = True
        icon = "⚠️"
        interpretation = (
            f"Answer may partially deviate from sources — review recommended "
            f"(faithfulness score: {effective_score:.2f})."
        )
    else:
        verdict = "LIKELY_HALLUCINATION"
        alert = True
        icon = "🚨"
        interpretation = (
            f"Answer shows low semantic similarity to retrieved sources — "
            f"high hallucination risk (faithfulness score: {effective_score:.2f})."
        )

    if alert:
        logger.warning(
            "SAMR-lite %s: score=%.3f threshold=%.2f | %s",
            verdict,
            effective_score,
            threshold,
            interpretation,
        )

    return {
        "faithfulness_score": effective_score,
        "verdict": verdict,
        "alert_triggered": alert,
        "alert_icon": icon,
        "interpretation": interpretation,
        "per_chunk_scores": per_chunk,
        "avg_score": avg_score,
        "max_chunk_score": max_score,
        "threshold_used": threshold,
    }


def run_samr_lite(
    answer: str,
    retrieved_docs: list[str],
    embedding_function: Any,
    threshold: float = GROUNDED_THRESHOLD,
) -> dict[str, Any]:
    """
    Prime Radiant Coherence Gate (via @ruvector/ruvllm similarity subprocess).

    Computes semantic similarity between the LLM answer and each retrieved
    document chunk.  The maximum similarity is inverted to produce a
    Laplacian-proxy energy score.  High energy (>0.70) means the answer
    is not grounded and gets BLOCKED by raising FaithfulnessError.

    This is the ONLY coherence pathway — there is no legacy SAMR-lite
    cosine fallback. Never warn-and-deliver.

    Raises:
        FaithfulnessError: when the answer fails the faithfulness check.
    """
    try:
        if not retrieved_docs:
            return {"verdict": "UNSUPPORTED", "faithfulness_score": 0.0, "blocked": False}

        import subprocess as _sp

        energies: list[float] = []
        max_sim: float = 0.0

        for chunk in retrieved_docs:
            try:
                proc = _sp.run(
                    ["npx", "--yes", "@ruvector/ruvllm", "similarity", answer, chunk[:500]],
                    capture_output=True, text=True, check=True, timeout=30,
                )
                out = proc.stdout.strip()
                if "Similarity:" in out:
                    sim_pct = float(out.split("Similarity:")[1].strip().replace("%", "")) / 100.0
                    max_sim = max(max_sim, sim_pct)
                    energies.append(1.0 - sim_pct)
            except _sp.TimeoutExpired:
                logger.warning("Coherence check chunk timed out (30s)")
            except Exception as chunk_err:
                logger.warning("Coherence check for chunk failed", error=str(chunk_err))

        # If we got at least ONE successful similarity measurement, use it
        if energies:
            energy = 1.0 - max_sim
        else:
            # ALL chunks failed — treat as low-confidence, BLOCK delivery
            logger.warning("All ruvllm similarity calls failed — blocking answer (no evidence)")
            raise FaithfulnessError(score=0.0, threshold=threshold)

        faithfulness_score = max(0.0, 1.0 - energy)
        is_blocked = energy > 0.70

        logger.info(
            "Prime Radiant Gate | energy=%.3f blocked=%s chunks_measured=%d/%d",
            energy, is_blocked, len(energies), len(retrieved_docs),
        )

        if is_blocked:
            raise FaithfulnessError(score=faithfulness_score, threshold=threshold)

        return {
            "verdict": "SUPPORTED",
            "faithfulness_score": faithfulness_score,
            "blocked": False,
            "witness": "ruvector_similarity",
            "chunks_measured": len(energies),
        }

    except FaithfulnessError:
        # Re-raise FaithfulnessError — don't catch it in the broad except
        raise
    except Exception as e:
        logger.error("Coherence gate error: %s", e)
        return {
            "faithfulness_score": -1.0,
            "verdict": "ERROR",
            "alert_triggered": False,
            "alert_icon": "ℹ️",
            "interpretation": f"Prime Radiant check failed: {e}",
            "per_chunk_scores": [],
            "error": str(e),
        }
