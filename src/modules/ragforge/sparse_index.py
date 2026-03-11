# AetherForge v1.0 — src/modules/ragforge/sparse_index.py
# ─────────────────────────────────────────────────────────────────
# FTS5 Sparse Index with BM25 Scoring
#
# Provides keyword-based retrieval using SQLite FTS5 virtual tables.
# Designed to complement ChromaDB's dense vector search.
#
# Architecture:
#   - SQLite FTS5 virtual table with BM25 ranking
#   - Reciprocal Rank Fusion (RRF) to merge dense + sparse results
#   - Zero external dependencies (SQLite is in Python stdlib)
#   - Shared data directory with ChromaDB (./data/chroma/)
#
# 100% offline. No internet required.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import structlog
import sqlite3
import threading
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = structlog.get_logger("aetherforge.ragforge.sparse")

# Default RRF constant — controls how much rank position matters
# Lower k = top-ranked results dominate more
RRF_K = 60


class SparseIndex:
    """
    FTS5-backed sparse keyword index with BM25 scoring.

    Each document chunk is stored with its full text and metadata.
    BM25 scoring handles term frequency, inverse document frequency,
    and document length normalization automatically via FTS5.
    """

    def __init__(self, db_path: str | Path = "./data/sparse_index.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()  # Serialize concurrent INSERT/COMMIT
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads
        assert self._conn is not None
        return self._conn

    def _ensure_schema(self) -> None:
        """Create FTS5 virtual table if it doesn't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                metadata   TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content_rowid='rowid',
                tokenize='porter unicode61'
            );

            -- Triggers to keep FTS5 index in sync with chunks table
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;
        """)
        conn.commit()
        logger.info("FTS5 sparse index ready at %s", self.db_path)

    def add_documents(self, chunks: list[Document]) -> int:
        """
        Add document chunks to the FTS5 index.
        Should be called alongside ChromaDB's add_documents().
        """
        if not chunks:
            return 0

        conn = self._get_conn()
        added = 0
        with self._write_lock:  # Prevent concurrent write corruption
            for chunk in chunks:
                chunk_id = chunk.metadata.get("chunk_id", "")
                if not chunk_id:
                    continue

                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO chunks (chunk_id, content, metadata) VALUES (?, ?, ?)",
                        (chunk_id, chunk.page_content, json.dumps(chunk.metadata)),
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    pass  # duplicate chunk_id, skip

            conn.commit()
        logger.info("FTS5: indexed %d new chunks", added)
        logger.debug("FTS5 total chunk count: %d", self.count())  # count() is O(n) — debug only
        return added

    def search(
        self,
        query: str,
        k: int = 10,
        source_filter: str | list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        BM25-scored keyword search using FTS5.

        Returns:
            List of (Document, bm25_score) tuples, sorted by relevance.
            BM25 scores are negative (FTS5 convention) — more negative = more relevant.
        """
        if not query.strip():
            return []

        conn = self._get_conn()

        # Sanitize query for FTS5 — remove special chars that break the parser
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        try:
            # BM25 scoring with FTS5
            # bm25(chunks_fts) returns negative values; more negative = better match
            rows = conn.execute("""
                SELECT c.chunk_id, c.content, c.metadata, bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (safe_query, k * 2)).fetchall()  # fetch extra for post-filtering
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query failed: %s (query: '%s')", e, safe_query[:100])
            return []

        results: list[tuple[Document, float]] = []
        for chunk_id, content, meta_json, score in rows:
            meta = json.loads(meta_json)

            # Apply source filter
            if source_filter:
                src = meta.get("source", "")
                if isinstance(source_filter, str) and src != source_filter:
                    continue
                if isinstance(source_filter, list) and src not in source_filter:
                    continue

            doc = Document(page_content=content, metadata=meta)
            results.append((doc, score))

            if len(results) >= k:
                break

        logger.info("FTS5/BM25: %d results for '%s...'", len(results), query[:50])
        return results

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source file."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE json_extract(metadata, '$.source') = ?",
                (source,),
            )
            conn.commit()
        deleted = cursor.rowcount
        logger.info("FTS5: deleted %d chunks from source '%s'", deleted, source)
        return deleted

    def get_vlm_chunks(self, source: str) -> list[Document]:
        """Return all VLM visual analysis chunks for a given source file.
        
        Used by the tiered VLM preservation logic to determine what tier
        of visual analysis has already been indexed for this document.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT content, metadata FROM chunks "
            "WHERE json_extract(metadata, '$.source') = ? "
            "AND json_extract(metadata, '$.chunk_type') = 'vlm_analysis'",
            (source,),
        ).fetchall()
        docs = []
        for content, meta_str in rows:
            try:
                meta = json.loads(meta_str)
            except Exception:
                meta = {}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def delete_vlm_chunks(self, source: str) -> int:
        """Delete only VLM visual analysis chunks for a source (preserves text chunks).
        
        Called when a higher-tier VLM is about to replace lower-tier visual analysis.
        Text chunks extracted by Docling are never touched.
        """
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.execute(
                "DELETE FROM chunks "
                "WHERE json_extract(metadata, '$.source') = ? "
                "AND json_extract(metadata, '$.chunk_type') = 'vlm_analysis'",
                (source,),
            )
            conn.commit()
        deleted = cursor.rowcount
        logger.info("FTS5: deleted %d VLM chunks for '%s'", deleted, source)
        return deleted

    def count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        Clean user query for FTS5 MATCH syntax.
        Removes special operators and wraps tokens for safe matching.
        """
        import re
        # Remove FTS5 special characters
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        # Split into tokens, remove empty
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        if not tokens:
            return ""
        # Join with implicit AND (FTS5 default)
        return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────
# Hybrid Search: Dense (ChromaDB) + Sparse (FTS5/BM25) via RRF
# ─────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    vector_store: Any,
    sparse_index: SparseIndex,
    k: int = 8,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    source_filter: str | list[str] | None = None,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) hybrid search.

    Combines:
      - Dense retrieval (ChromaDB + BGE-M3 cosine similarity)
      - Sparse retrieval (SQLite FTS5 + BM25 keyword scoring)

    Returns deduplicated, re-ranked documents.

    RRF formula per doc:
        score = dense_weight / (k + dense_rank) + sparse_weight / (k + sparse_rank)

    Args:
        k: Number of results to return
        dense_weight: Weight for semantic similarity (default: 0.6)
        sparse_weight: Weight for keyword relevance (default: 0.4)
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    dense_count = 0
    sparse_count = 0

    # ── Dense retrieval (ChromaDB) ────────────────────────────────
    filter_kwargs = {}
    if source_filter:
        if isinstance(source_filter, str):
            filter_kwargs["filter"] = {"source": source_filter}
        elif isinstance(source_filter, list) and len(source_filter) == 1:
            filter_kwargs["filter"] = {"source": source_filter[0]}
        elif isinstance(source_filter, list):
            filter_kwargs["filter"] = {"source": {"$in": source_filter}}

    try:
        dense_results = vector_store.similarity_search(query, k=k * 2, **filter_kwargs)
        dense_count = len(dense_results)
        for rank, doc in enumerate(dense_results):
            doc_id = _doc_identifier(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + dense_weight / (RRF_K + rank + 1)
            doc_map[doc_id] = doc
        logger.info("Dense retrieval: %d results for '%s...'", dense_count, query[:50])
    except Exception as e:
        logger.warning("Dense search failed: %s", e, exc_info=True)

    # ── Sparse retrieval (FTS5/BM25) ─────────────────────────────
    try:
        sparse_results = sparse_index.search(
            query, k=k * 2, source_filter=source_filter,
        )
        sparse_count = len(sparse_results)
        for rank, (doc, _bm25_score) in enumerate(sparse_results):
            doc_id = _doc_identifier(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + sparse_weight / (RRF_K + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        logger.info("Sparse retrieval: %d results for '%s...'", sparse_count, query[:50])
    except Exception as e:
        logger.warning("Sparse search failed: %s — falling back to dense-only", e, exc_info=True)

    # ── Rank by combined RRF score ────────────────────────────────
    ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    final_docs = [doc_map[doc_id] for doc_id in ranked_ids[:k]]

    logger.info(
        "Hybrid search: %d final results (dense=%d, sparse=%d, fused=%d)",
        len(final_docs), dense_count, sparse_count, len(rrf_scores),
    )
    return final_docs


def _doc_identifier(doc: Document) -> str:
    """Create a stable identifier for deduplication across dense/sparse results."""
    meta = doc.metadata
    chunk_id = meta.get("chunk_id", "")
    if chunk_id:
        return chunk_id
    # Fallback: hash of source + page + content prefix
    source = meta.get("source", "")
    page = str(meta.get("page", ""))
    content_prefix = doc.page_content[:100]
    return f"{source}:{page}:{hash(content_prefix)}"
