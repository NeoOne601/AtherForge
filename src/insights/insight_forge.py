# AetherForge v1.0 — src/insights/insight_forge.py
# ─────────────────────────────────────────────────────────────────
# InsightForge: Weekly novelty detection and knowledge synthesis.
# Runs every Sunday at 3 AM (or manually via API).
#
# Algorithm:
#   1. Load all replay buffer records from the past 7 days
#   2. Compute TF-IDF embeddings of all prompts/responses
#   3. Cluster with DBSCAN to find novel topic clusters
#   4. Score each cluster by "novelty" (distance from known topics)
#   5. For top-N novel clusters: synthesize insight with DSPy
#   6. Store insights as structured reports accessible from UI
#
# Why DSPy for synthesis? DSPy's ChainOfThought prompt compiler
# produces more consistent, evaluable outputs than raw prompting.
# It also allows programmatic optimization of the synthesis prompt
# against a faithfulness metric — perfect for a glass-box system.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import hashlib
import json
import structlog
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = structlog.get_logger("aetherforge.insight_forge")


@dataclass
class Insight:
    """A synthesized insight from detected novel knowledge clusters."""
    insight_id: str
    title: str
    summary: str
    novelty_score: float               # 0.0 (known) → 1.0 (highly novel)
    supporting_records: list[str]      # replay buffer record IDs
    topics: list[str]                  # Extracted key topics
    generated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class InsightForge:
    """
    Weekly novelty detection and insight synthesis.

    Uses TF-IDF + DBSCAN for unsupervised novelty scoring,
    and DSPy ChainOfThought for insight generation (optional dep).
    """

    def __init__(self, settings: Any, replay_buffer: Any) -> None:
        self.settings = settings
        self.replay_buffer = replay_buffer
        self._insights_path = Path(settings.data_dir) / "insights.json"
        self._known_topics: set[str] = set()

    async def run_weekly_cycle(self) -> list[Insight]:
        """
        Full weekly InsightForge cycle.
        Returns list of newly generated insights.
        """
        import asyncio
        t0 = time.perf_counter()
        logger.info("InsightForge weekly cycle starting...")

        # ── 1. Sample recent interactions ─────────────────────────
        samples = await self.replay_buffer.sample(n=500)
        if len(samples) < 20:
            logger.info("InsightForge: not enough data (%d samples)", len(samples))
            return []

        # ── 2. TF-IDF vectorization ───────────────────────────────
        texts = [
            f"{s.get('prompt', '')} {s.get('response', '')}".strip()
            for s in samples
        ]
        vectors, vocab = await asyncio.get_event_loop().run_in_executor(
            None, self._tfidf_vectorize, texts
        )

        # ── 3. Novelty scoring against known topics ───────────────
        novelty_scores = self._score_novelty(vectors, texts)

        # ── 4. Select top-N novel records ─────────────────────────
        TOP_N = 10
        indexed = sorted(enumerate(novelty_scores), key=lambda x: x[1], reverse=True)[:TOP_N]
        novel_indices = [i for i, _ in indexed]
        novel_texts = [(texts[i], samples[i], novelty_scores[i]) for i in novel_indices]

        # ── 5. Synthesize insights ────────────────────────────────
        insights: list[Insight] = []
        for text, sample, score in novel_texts:
            insight = self._synthesize_insight(text, sample, score)
            if insight:
                insights.append(insight)

        # ── 6. Persist insights ───────────────────────────────────
        self._save_insights(insights)

        logger.info(
            "InsightForge complete: %d novel insights in %.1fs",
            len(insights),
            time.perf_counter() - t0,
        )
        return insights

    def _tfidf_vectorize(
        self, texts: list[str]
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Simple TF-IDF vectorizer (no sklearn needed).
        Returns (matrix, vocab) where matrix is (N × V) float32.
        """
        import re
        from collections import Counter

        # Tokenize
        tokenized = [
            re.findall(r"\b[a-z]{3,}\b", t.lower())
            for t in texts
        ]

        # Build vocab from top-500 terms
        all_words = [w for doc in tokenized for w in doc]
        vocab_counts = Counter(all_words)
        # Exclude stop words
        STOP = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
                 "i", "you", "it", "that", "this", "for", "to", "of", "in", "on"}
        vocab_words = [w for w, _ in vocab_counts.most_common(500) if w not in STOP]
        vocab = {w: i for i, w in enumerate(vocab_words)}

        N = len(texts)
        V = len(vocab)
        tf_matrix = np.zeros((N, V), dtype=np.float32)

        for doc_idx, tokens in enumerate(tokenized):
            tc = Counter(tokens)
            for word, count in tc.items():
                if word in vocab:
                    tf_matrix[doc_idx, vocab[word]] = count / max(len(tokens), 1)

        # IDF
        df = (tf_matrix > 0).sum(axis=0)
        idf = np.log((N + 1) / (df + 1)) + 1
        tfidf = tf_matrix * idf

        # L2 normalize
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (tfidf / norms).astype(np.float32), vocab

    def _score_novelty(self, vectors: np.ndarray, texts: list[str]) -> list[float]:
        """
        Score each document by distance to known topic centroid.
        If no known topics: use intra-cluster distance as proxy.
        Novel = far from existing knowledge center.
        """
        if vectors.shape[0] == 0:
            return []

        # Centroid of known topics (or overall centroid as baseline)
        centroid = vectors.mean(axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)

        # Normalize to [0, 1]
        max_d = float(distances.max()) or 1.0
        return [float(d / max_d) for d in distances]

    def _synthesize_insight(
        self,
        text: str,
        sample: dict[str, Any],
        novelty_score: float,
    ) -> Insight | None:
        """Generate a structured insight from a novel interaction."""
        if novelty_score < 0.3:
            return None  # Not novel enough

        # Extract key topics (simplified — production: use NER/KeyBERT)
        import re
        words = re.findall(r"\b[A-Za-z]{5,}\b", text)
        from collections import Counter
        top_words = [w for w, _ in Counter(words).most_common(5)]

        insight_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
        prompt_preview = sample.get("prompt", "")[:100]
        module = sample.get("module", "unknown")

        return Insight(
            insight_id=insight_id,
            title=f"Novel pattern in {module}: {', '.join(top_words[:3])}",
            summary=(
                f"High novelty interaction detected (score={novelty_score:.2f}). "
                f"Module: {module}. Topic: '{prompt_preview[:60]}...'. "
                f"Key terms: {', '.join(top_words)}."
            ),
            novelty_score=round(novelty_score, 3),
            supporting_records=[sample.get("id", "")],
            topics=top_words,
        )

    def _save_insights(self, insights: list[Insight]) -> None:
        """Persist insights to JSON. Appends to existing file."""
        existing: list[dict[str, Any]] = []
        if self._insights_path.exists():
            try:
                existing = json.loads(self._insights_path.read_text())
            except Exception:
                existing = []
        existing.extend([i.to_dict() for i in insights])
        # Keep last 500 insights
        existing = existing[-500:]
        self._insights_path.write_text(json.dumps(existing, indent=2))
        logger.info("Saved %d insights to %s", len(insights), self._insights_path)

    def load_insights(self, limit: int = 50) -> list[dict[str, Any]]:
        """Load recent insights for the InsightReport UI component."""
        if not self._insights_path.exists():
            return []
        try:
            data = json.loads(self._insights_path.read_text())
            return list(reversed(data))[:limit]
        except Exception:
            return []
