# AetherForge v1.0 — src/modules/ragforge/graph.py
# ─────────────────────────────────────────────────────────────────
# RAGForge: Retrieval-Augmented Generation module.
# Uses ChromaDB (embedded, no server) for local vector storage.
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
    n_results: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Main RAGForge execution: embed query → retrieve → return context.

    Returns:
      {"documents": [...], "distances": [...], "ids": [...]}
    """
    try:
        col = _get_collection(chroma_path)
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        logger.info("RAGForge retrieved %d docs for query (len=%d)", len(docs), len(query))
        return {
            "documents": docs,
            "distances": distances,
            "metadatas": metadatas,
            "count": len(docs),
        }
    except Exception as exc:
        logger.exception("RAGForge query failed: %s", exc)
        return {"documents": [], "distances": [], "metadatas": [], "count": 0, "error": str(exc)}


def ingest_documents(
    texts: list[str],
    ids: list[str] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
    chroma_path: str = "./data/chroma",
) -> dict[str, Any]:
    """
    Ingest documents into ChromaDB.
    ChromaDB handles embedding via its default sentence-transformers model.
    For air-gapped: set CHROMA_EMBEDDING_MODEL to a local path.
    """
    import uuid as uuid_mod
    col = _get_collection(chroma_path)

    if ids is None:
        ids = [str(uuid_mod.uuid4()) for _ in texts]
    if metadatas is None:
        metadatas = [{} for _ in texts]

    col.add(documents=texts, ids=ids, metadatas=metadatas)
    logger.info("Ingested %d documents into RAGForge", len(texts))
    return {"added": len(texts), "total": col.count()}
