# AetherForge v1.0 — src/modules/ragforge/benchmarker.py
# ─────────────────────────────────────────────────────────────────
# RAG Performance Benchmarking Suite.
#
# Runs a fixed 'Golden Set' of queries to measure RAG grounding
# and faithfulness using the current system parameters.
# ─────────────────────────────────────────────────────────────────

import asyncio
import json
import os
from typing import Any

import structlog

from src.modules.ragforge.cognitive_rag import CognitiveRAG
from src.modules.ragforge.history_manager import RAGHistoryManager
from src.modules.ragforge.samr_lite import run_samr_lite

logger = structlog.get_logger("aetherforge.rag_benchmarker")


class RAGBenchmarker:
    """
    Standardized benchmark for RAG performance evaluation.
    """

    def __init__(
        self, rag: CognitiveRAG, history: RAGHistoryManager, embeddings: Any = None
    ) -> None:
        self.rag = rag
        self.history = history
        self.embeddings = embeddings
        self.dataset_path = "data/benchmark_dataset.json"

    async def run_suite(self) -> float:
        """
        Runs the benchmark dataset and returns the average combined score.
        """
        if not os.path.exists(self.dataset_path):
            logger.warning(f"Benchmark dataset {self.dataset_path} not found. using fallback.")
            queries = [{"query": "What is AetherForge?", "context": "", "expected_facts": []}]
        else:
            with open(self.dataset_path) as f:
                queries = json.load(f)

        logger.info("Starting RAG Benchmark Suite (%d queries)...", len(queries))
        scores = []

        for q_obj in queries:
            query = q_obj.get("query", "")
            if not query:
                continue
            try:
                # No extra retries for benchmarks to ensure raw performance measurement
                answer, docs, trace = self.rag.think_and_answer(query, max_retries=0)

                score = trace.grounding_score

                if self.embeddings and docs:
                    doc_texts = [getattr(d, "page_content", str(d)) for d in docs]
                    samr_res = run_samr_lite(answer, doc_texts, self.embeddings)
                    faithfulness = samr_res.get("faithfulness_score", 0.0)
                    score = (score * 0.5) + (faithfulness * 0.5)

                scores.append(score)
                # Small sleep to prevent token saturation on edge devices
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error("Benchmark query failed '%s': %s", query, e)
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info("RAG Benchmark complete. Average Combined Score: %.4f", avg_score)
        return avg_score
