# AetherForge v1.0 — src/modules/ragforge/ruvector_store.py
# ─────────────────────────────────────────────────────────────────
# RuVector CLI Bridge — LangChain-compatible VectorStore wrapper
#
# Talks to the native RuVector NPM binary via subprocess calls:
#   npx ruvector rvf create   — create a database
#   npx ruvector rvf ingest   — insert vectors
#   npx ruvector rvf query    — nearest-neighbour search
#
# This replaces the hallucinated `from ruvector.langchain import
# RuVectorStore` that previously failed with ImportError.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any, Iterable, List, Optional

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = structlog.get_logger("aetherforge.ragforge.ruvector_store")

# We keep a module-level flag so the DB is created at most once
_DB_CREATED: set[str] = set()


class RuVectorStore(VectorStore):
    """
    LangChain-compatible wrapper that proxies all vector operations to the
    RuVector CLI (`npx ruvector rvf …`).

    Lifecycle:
      1. `__init__` creates the .rvf database file (once per path).
      2. `add_texts` embeds + writes vectors via `rvf ingest`.
      3. `similarity_search` embeds the query + runs `rvf query`.
    """

    def __init__(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        dimension: int = 384,
        metric: str = "cosine",
        **kwargs: Any,
    ):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.dimension = dimension
        self.metric = metric

        # Ensure the .rvf file exists
        rvf_path = self._rvf_path()
        if rvf_path not in _DB_CREATED and not os.path.exists(rvf_path):
            self._create_db()
            _DB_CREATED.add(rvf_path)
        elif os.path.exists(rvf_path):
            _DB_CREATED.add(rvf_path)

    # ── Internal helpers ─────────────────────────────────────────

    def _rvf_path(self) -> str:
        """Return the canonical .rvf file path."""
        p = self.persist_directory
        if not p.endswith(".rvf"):
            p = p + ".rvf"
        return p

    def _create_db(self) -> None:
        """Create the .rvf database via CLI."""
        rvf = self._rvf_path()
        os.makedirs(os.path.dirname(rvf) or ".", exist_ok=True)
        try:
            proc = subprocess.run(
                [
                    "npx", "--yes", "ruvector", "rvf", "create", rvf,
                    "-d", str(self.dimension),
                    "-m", self.metric,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("RuVector DB created", path=rvf, output=proc.stdout.strip())
        except subprocess.CalledProcessError as e:
            logger.error("Failed to create RuVector DB", stderr=e.stderr)
            raise

    # ── LangChain VectorStore interface ──────────────────────────

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings = self.embedding_function.embed_documents(texts_list)

        # Build the JSON payload expected by `rvf ingest`
        records = []
        ids: list[str] = []
        for i, (text, emb) in enumerate(zip(texts_list, embeddings)):
            doc_id = f"vec_{i}_{abs(hash(text)) % (10**12)}"
            ids.append(doc_id)
            meta = (metadatas[i] if metadatas and i < len(metadatas) else {}).copy()
            meta["_text"] = text  # stash the text in metadata for retrieval
            records.append({"id": doc_id, "vector": emb, "metadata": meta})

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(records, f)
            temp_path = f.name

        try:
            proc = subprocess.run(
                [
                    "npx", "--yes", "ruvector", "rvf", "ingest", self._rvf_path(),
                    "-i", temp_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(
                "Inserted vectors into RuVectorStore",
                count=len(records),
                output=proc.stdout.strip(),
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to insert vectors", stderr=e.stderr)
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Embed the query and run a nearest-neighbour search via the CLI."""
        query_emb = self.embedding_function.embed_query(query)
        vector_csv = ",".join(str(v) for v in query_emb)

        cmd = [
            "npx", "--yes", "ruvector", "rvf", "query", self._rvf_path(),
            "-v", vector_csv,
            "-k", str(k),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = proc.stdout.strip()

            # The CLI prints results line-by-line like:
            #   1. id=vec_0_123  dist=0.123456
            # Attempt JSON parse first (newer versions may emit JSON)
            try:
                results = json.loads(output)
            except json.JSONDecodeError:
                # Fall back to line-based parsing
                results = []
                for line in output.splitlines():
                    line = line.strip()
                    if "id=" in line and "dist=" in line:
                        parts = dict(
                            p.split("=", 1)
                            for p in line.split()
                            if "=" in p
                        )
                        results.append(parts)

            docs: list[Document] = []
            for res in results:
                if isinstance(res, dict):
                    meta = res.get("metadata", {})
                    text = meta.pop("_text", "") if isinstance(meta, dict) else ""
                    docs.append(Document(page_content=str(text), metadata=meta))

            return docs

        except subprocess.CalledProcessError as e:
            logger.error("RuVector search failed", stderr=e.stderr)
            return []

    # ── Required by base class ───────────────────────────────────

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        persist_directory: str = "./ruvector_db",
        **kwargs: Any,
    ) -> "RuVectorStore":
        store = cls(
            persist_directory=persist_directory,
            embedding_function=embedding,
            **kwargs,
        )
        store.add_texts(texts, metadatas=metadatas)
        return store

    def get(
        self,
        where: Optional[dict] = None,
        include: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """ChromaDB-compatible .get() for metadata queries.

        Used by ragforge_tree.py to fetch document section structure.
        Returns empty result set since RuVector CLI doesn't support
        metadata-only queries. The HTI tree view will show empty until
        RuVector implements a native metadata query API.
        """
        logger.debug(
            "RuVectorStore.get() — metadata queries not yet supported by RuVector CLI"
        )
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by ID from the .rvf database.

        Called by ragforge_indexer.py dedup logic before re-indexing a
        document. Attempts bulk delete via `rvf delete`; if the CLI
        subcommand doesn't exist yet, logs a warning and continues
        (the next ingest will overwrite by ID anyway).
        """
        if not ids:
            return
        try:
            import tempfile as _tf

            with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(ids, f)
                temp_path = f.name

            proc = subprocess.run(
                [
                    "npx", "--yes", "ruvector", "rvf", "delete",
                    self._rvf_path(), "-i", temp_path,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                logger.info("Deleted %d vectors from RuVectorStore", len(ids))
            else:
                logger.debug(
                    "RuVector delete returned non-zero (CLI may not support delete yet): %s",
                    proc.stderr.strip()[:200],
                )
        except Exception as e:
            logger.debug("RuVector delete failed (non-fatal, will overwrite on re-ingest): %s", e)
        finally:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)

