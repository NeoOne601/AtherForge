# AetherForge v1.0 — src/modules/ragforge/graph.py
# ─────────────────────────────────────────────────────────────────
# RAGForge: Retrieval-Augmented Generation module.
# Uses ChromaDB (embedded, no server) for local vector storage.
# Embedding model: BAAI/bge-m3 (8192 tokens, 1024 dims)
# SAMR-lite: post-retrieval faithfulness check (adapted from Ventro)
# Every retrieval passes through Silicon Colosseum before execution.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("aetherforge.ragforge")

# ── ChromaDB client (lazy init) ───────────────────────────────────
_chroma_client: Any = None
_collection: Any = None


def _get_collection(chroma_path: str) -> Any:
    """Lazily initialize ChromaDB persistent client."""
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
        _collection = _chroma_client.get_or_create_collection(
            name="aetherforge_ragforge",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection ready: %d docs", _collection.count())
    return _collection


def build_ragforge_graph() -> dict[str, Any]:
    """
    Build and return the RAGForge module descriptor.
    Returns a dict with run() callable that the meta-agent invokes.
    """
    return {
        "module_id": "ragforge",
        "run": run_ragforge,
        "ingest": ingest_documents,
    }


def run_ragforge(
    query: str,
    chroma_path: str,
    n_results: int = 6,
    embedding_function: Any = None,
    samr_enabled: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Main RAGForge execution: embed query → retrieve → SAMR-lite → return.

    Retrieval uses BGE-M3's cosine similarity (dense retrieval).
    SAMR-lite performs post-retrieval faithfulness measurement — if the
    answer embedding diverges significantly from the retrieved context,
    the response is flagged for human review.

    Returns:
      {
        "documents": [...],     # retrieved text chunks
        "metadatas": [...],     # source, page, section, chunk_type
        "distances": [...],     # cosine distances (lower = more similar)
        "count": int,
        "samr": {...}           # SAMR-lite faithfulness verdict (if enabled)
      }
    """
    try:
        col = _get_collection(chroma_path)
        total_docs = col.count()
        if total_docs == 0:
            return {
                "documents": [], "distances": [], "metadatas": [],
                "count": 0, "samr": None,
                "message": "Knowledge base is empty. Upload documents first."
            }

        results = col.query(
            query_texts=[query],
            n_results=min(n_results, total_docs),
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        logger.info(
            "RAGForge retrieved %d chunks for query (len=%d) | top_distance=%.3f",
            len(docs), len(query), min(distances) if distances else -1
        )

        # ── SAMR-lite: faithfulness check ─────────────────────────
        # Note: SAMR-lite requires the answer embedding, not just the
        # retrieved context. The meta-agent calls run_samr_check()
        # AFTER the LLM generates its answer and passes it back here.
        # This function returns the retrieved context. The meta-agent
        # orchestrates the SAMR check as a separate step.
        # ─────────────────────────────────────────────────────────

        return {
            "documents": docs,
            "distances": distances,
            "metadatas": metadatas,
            "count": len(docs),
            "samr": None,  # populated by meta-agent after LLM generates answer
        }
    except Exception as exc:
        logger.exception("RAGForge query failed: %s", exc)
        return {"documents": [], "distances": [], "metadatas": [], "count": 0, "error": str(exc)}


def run_samr_check(
    answer: str,
    retrieved_docs: list[str],
    embedding_function: Any,
) -> dict[str, Any]:
    """
    Run SAMR-lite faithfulness check after the LLM has generated its answer.
    Called by the meta-agent post-generation.

    Returns the SAMR verdict dict:
      {
        "faithfulness_score": float,
        "verdict": "GROUNDED" | "LOW_CONFIDENCE" | "LIKELY_HALLUCINATION",
        "alert_triggered": bool,
        "alert_icon": str,
        "interpretation": str,
      }
    """
    if not embedding_function:
        logger.debug("SAMR-lite: no embedding function provided, skipping")
        return {"verdict": "SKIPPED", "alert_triggered": False}

    try:
        from src.modules.ragforge.samr_lite import run_samr_lite
        return run_samr_lite(
            answer=answer,
            retrieved_docs=retrieved_docs,
            embedding_function=embedding_function,
        )
    except Exception as e:
        logger.error("SAMR-lite call failed: %s", e)
        return {"verdict": "ERROR", "alert_triggered": False, "error": str(e)}


def ingest_documents(
    texts: list[str],
    ids: list[str] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
    chroma_path: str = "./data/chroma",
) -> dict[str, Any]:
    """
    Ingest pre-processed text chunks into ChromaDB.
    Used by the meta-agent tools and the upload API.
    For file-based ingestion, use ragforge_indexer.index_document() instead.
    """
    import uuid as uuid_mod
    col = _get_collection(chroma_path)

    if ids is None:
        ids = [str(uuid_mod.uuid4()) for _ in texts]
    if metadatas is None:
        metadatas = [{"source": "api", "chunk_type": "section"} for _ in texts]

    col.add(documents=texts, ids=ids, metadatas=metadatas)
    logger.info("Ingested %d documents into RAGForge | total=%d", len(texts), col.count())
    return {"added": len(texts), "total": col.count()}
