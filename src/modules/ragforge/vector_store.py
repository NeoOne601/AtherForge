# AetherForge v1.0 — src/modules/ragforge/vector_store.py
# ─────────────────────────────────────────────────────────────────
# RuVectorStore — GNN-HNSW hybrid search wrapper.
# Replaces ChromaDB. Uses RuVector's GNN-HNSW with hybrid search.
# Hybrid = 70% dense semantic + 30% BM25 sparse keyword search.
# GNN reranking improves results over time as queries accumulate.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RuVectorStore:
    """
    High-level async wrapper for RuVector GNN-HNSW hybrid search.

    This is the spec-defined interface from SA-04.
    The actual runtime implementation is in `ruvector_store.py` (CLI bridge)
    which implements the LangChain VectorStore protocol.

    This class provides the simplified async interface used by new code
    that doesn't need LangChain compatibility.
    """

    def __init__(self, dimensions: int = 1024):
        """dimensions=1024 matches BAAI/bge-m3 embedding model."""
        self.dimensions = dimensions
        self._initialized = False
        logger.info("RuVectorStore initialised: dim=%d gnn=True", dimensions)

    async def search(
        self,
        embedding: list[float],
        top_k: int = 12,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid semantic + keyword search with GNN reranking.
        Returns list of dicts with keys: id, text, metadata, score.
        """
        try:
            from ruvector import RuVectorClient  # type: ignore
            client = RuVectorClient(dimensions=self.dimensions, gnn=True)
            return await client.hybrid_search(
                query=embedding,
                k=top_k,
                hybrid_alpha=0.7,    # 70% semantic, 30% BM25
                rerank=True,         # GNN layer reranks candidates
                where=where,
            )
        except ImportError:
            logger.warning("ruvector package not available; search returns empty")
            return []

    async def add(
        self,
        embedding: list[float],
        text: str,
        metadata: dict,
        doc_id: str | None = None,
    ) -> str:
        """Add a document chunk. Returns the stored document ID."""
        try:
            from ruvector import RuVectorClient  # type: ignore
            client = RuVectorClient(dimensions=self.dimensions, gnn=True)
            return await client.upsert(
                embedding=embedding,
                text=text,
                metadata=metadata,
                id=doc_id,
            )
        except ImportError:
            logger.warning("ruvector package not available; add skipped")
            return doc_id or "noop"

    async def add_batch(self, items: list[dict]) -> list[str]:
        """Batch insert for ingestion efficiency."""
        try:
            from ruvector import RuVectorClient  # type: ignore
            client = RuVectorClient(dimensions=self.dimensions, gnn=True)
            return await client.upsert_batch(items)
        except ImportError:
            return [item.get("id", "noop") for item in items]

    async def delete(self, doc_id: str) -> None:
        try:
            from ruvector import RuVectorClient  # type: ignore
            client = RuVectorClient(dimensions=self.dimensions, gnn=True)
            await client.delete(doc_id)
        except ImportError:
            pass

    async def count(self) -> int:
        try:
            from ruvector import RuVectorClient  # type: ignore
            client = RuVectorClient(dimensions=self.dimensions, gnn=True)
            return await client.count()
        except ImportError:
            return 0

    async def reset(self) -> None:
        """Clear all vectors. Used in tests only."""
        try:
            from ruvector import RuVectorClient  # type: ignore
            client = RuVectorClient(dimensions=self.dimensions, gnn=True)
            await client.reset()
        except ImportError:
            pass
