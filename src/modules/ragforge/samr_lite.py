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
#   < 0.55 → LOW_CONFIDENCE (answer may contain hallucinations)
#   < 0.30 → LIKELY_HALLUCINATION (answer barely relates to sources)
#
# Note: CognitiveRAG produces chain-of-thought synthesized answers that
# naturally have lower cosine similarity to raw chunks than verbatim
# retrieval. The thresholds are tuned accordingly.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import structlog
import math
from typing import Any

logger = structlog.get_logger("aetherforge.ragforge.samr_lite")

# Thresholds (tuned for CognitiveRAG chain-of-thought synthesized answers)
GROUNDED_THRESHOLD = 0.55       # answer is grounded in sources
LOW_CONFIDENCE_THRESHOLD = 0.30  # answer is questionable
DEFAULT_DIMS = 384              # all-MiniLM-L6-v2 embedding dimensions


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
        round(_cosine_similarity(answer_embedding, ctx_emb), 4)
        for ctx_emb in context_embeddings
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
            verdict, effective_score, threshold, interpretation
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
    High-level SAMR-lite entry point.

    Embeds the answer and retrieved documents using the provided
    embedding function (BGE-M3 via LangChain's HuggingFaceEmbeddings),
    then computes the faithfulness verdict.

    Args:
        answer:             The LLM-generated answer string
        retrieved_docs:     List of source document chunk texts
        embedding_function: LangChain embedding model (all-MiniLM-L6-v2)
        threshold:          Faithfulness threshold (default: 0.55)
    """
    try:
        # Embed answer and all retrieved chunks
        answer_emb = embedding_function.embed_query(answer)
        if not retrieved_docs:
            return compute_faithfulness(answer_emb, [], threshold)

        context_embs = embedding_function.embed_documents(retrieved_docs)
        result = compute_faithfulness(answer_emb, context_embs, threshold)
        logger.debug(
            "SAMR-lite complete | verdict=%s score=%.3f chunks=%d",
            result["verdict"], result["faithfulness_score"], len(retrieved_docs)
        )
        return result

    except Exception as e:
        logger.error("SAMR-lite error: %s", e)
        return {
            "faithfulness_score": -1.0,
            "verdict": "ERROR",
            "alert_triggered": False,
            "alert_icon": "ℹ️",
            "interpretation": f"SAMR-lite check failed: {e}",
            "per_chunk_scores": [],
            "error": str(e),
        }
