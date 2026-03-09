# AetherForge v1.0 — src/modules/ragforge/benchmarker.py
# ─────────────────────────────────────────────────────────────────
# RAG Performance Benchmarking Suite.
#
# Runs a fixed 'Golden Set' of queries to measure RAG grounding
# and faithfulness using the current system parameters.
# ─────────────────────────────────────────────────────────────────

import asyncio
import structlog
import time
from typing import Dict, List

from src.modules.ragforge.cognitive_rag import CognitiveRAG
from src.modules.ragforge.history_manager import RAGHistoryManager

logger = structlog.get_logger("aetherforge.rag_benchmarker")

class RAGBenchmarker:
    """
    Standardized benchmark for RAG performance evaluation.
    """

    GOLDEN_QUERIES = [
        "What is AetherForge?",
        "Explain the IRA Framework.",
        "How does OPLoRA work?",
        "What are the key benefits of local-first AI?",
        "Summarize the BitNet architecture."
    ]

    def __init__(self, rag: CognitiveRAG, history: RAGHistoryManager) -> None:
        self.rag = rag
        self.history = history

    async def run_suite(self) -> float:
        """
        Runs the golden query set and returns the average grounding score.
        """
        logger.info("Starting RAG Benchmark Suite (%d queries)...", len(self.GOLDEN_QUERIES))
        scores = []
        
        for query in self.GOLDEN_QUERIES:
            try:
                # No extra retries for benchmarks to ensure raw performance measurement
                _, _, trace = self.rag.think_and_answer(query, max_retries=0)
                scores.append(trace.grounding_score)
                # Small sleep to prevent token saturation on edge devices
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error("Benchmark query failed '%s': %s", query, e)
                scores.append(0.0)
                
        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info("RAG Benchmark complete. Average Grounding Score: %.4f", avg_score)
        return avg_score
