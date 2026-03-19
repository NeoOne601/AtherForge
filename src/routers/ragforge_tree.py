# AetherForge v1.0 — src/routers/ragforge_tree.py
# ─────────────────────────────────────────────────────────────────
# Hierarchical Tree Index (HTI) Router
# Exposes the document section tree built by Docling HTI tagging.
# Frontend uses this to render a tree-browser HUD.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1/ragforge/tree", tags=["RAGForge Tree"])
logger = structlog.get_logger("aetherforge.ragforge_tree")


@router.get("/{source_name}")
async def get_document_tree(source_name: str, request: Request) -> JSONResponse:
    """
    Return the hierarchical section tree for a specific document.
    Reconstructs the tree from HTI metadata stored in ChromaDB.

    Response shape:
    {
        "source": "document.pdf",
        "tree": [
            {
                "id": "<section_id>",
                "label": "Chapter 1",
                "path": "Root / Chapter 1",
                "parent_id": "<parent_uuid>",
                "chunk_count": 5,
                "pages": [1, 2, 3],
                "children": [...]
            }
        ]
    }
    """
    state = request.app.state.app_state
    vector_store = state.vector_store

    try:
        # Fetch all chunks for this source with HTI metadata
        results = vector_store.get(
            where={"source": source_name},
            include=["metadatas"],
        )
        metadatas: list[dict[str, Any]] = results.get("metadatas", [])

        if not metadatas:
            return JSONResponse({"source": source_name, "tree": [], "total_chunks": 0})

        # Build section registry: section_id -> node info
        section_map: dict[str, dict] = {}
        root_ids: list[str] = []

        for meta in metadatas:
            sid = meta.get("section_id", "")
            pid = meta.get("parent_id")
            path = meta.get("path", "Root")
            section = meta.get("section", "Unknown")
            page = meta.get("page", 0)

            if not sid:
                continue

            if sid not in section_map:
                section_map[sid] = {
                    "id": sid,
                    "label": section,
                    "path": path,
                    "parent_id": pid,
                    "chunk_count": 0,
                    "pages": set(),
                    "children": [],
                }
            section_map[sid]["chunk_count"] += 1
            section_map[sid]["pages"].add(page)

            if not pid or pid not in section_map:
                if sid not in root_ids:
                    root_ids.append(sid)

        # Wire up children
        children_map: dict[str, list[str]] = defaultdict(list)
        for sid, node in section_map.items():
            pid = node.get("parent_id")
            if pid and pid in section_map:
                children_map[pid].append(sid)
                # Remove from root if it has a real parent
                if sid in root_ids:
                    root_ids.remove(sid)

        def build_node(sid: str, depth: int = 0) -> dict:
            node = section_map[sid]
            return {
                "id": sid,
                "label": node["label"],
                "path": node["path"],
                "parent_id": node["parent_id"],
                "chunk_count": node["chunk_count"],
                "pages": sorted(node["pages"]),
                "depth": depth,
                "children": [
                    build_node(child_id, depth + 1)
                    for child_id in children_map.get(sid, [])
                ],
            }

        tree = [build_node(rid) for rid in root_ids]
        total_chunks = sum(n["chunk_count"] for n in section_map.values())

        return JSONResponse({
            "source": source_name,
            "tree": tree,
            "total_sections": len(section_map),
            "total_chunks": total_chunks,
        })

    except Exception as exc:
        logger.exception("Failed to build HTI tree for '%s': %s", source_name, exc)
        return JSONResponse({"source": source_name, "tree": [], "error": str(exc)}, status_code=500)


@router.get("/{source_name}/section/{section_id}")
async def get_section_chunks(
    source_name: str, section_id: str, request: Request
) -> JSONResponse:
    """
    Return the actual text chunks belonging to a specific section node.
    Used by the frontend tree browser when a user clicks into a section.
    """
    state = request.app.state.app_state
    vector_store = state.vector_store

    try:
        results = vector_store.get(
            where={"$and": [{"source": source_name}, {"section_id": section_id}]},
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        chunks = [
            {"text": text, "meta": meta}
            for text, meta in zip(docs, metas)
        ]

        return JSONResponse({
            "source": source_name,
            "section_id": section_id,
            "chunks": chunks,
            "total": len(chunks),
        })

    except Exception as exc:
        logger.exception("Failed to fetch section chunks: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
