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
    sparse_index = state.sparse_index

    try:
        # Fetch all chunks for this source with HTI metadata via SQLite FTS
        docs = sparse_index.get_chunks_by_source(source_name)
        metadatas: list[dict[str, Any]] = [doc.metadata for doc in docs]

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


@router.get("/graph/{source_name}")
async def get_document_graph(source_name: str, request: Request) -> JSONResponse:
    """
    Generates a coherent node/edge graph from the document's structural metadata.
    Simulates @ruvector/graph-node traversing SONA clustered patterns.
    """
    state = request.app.state.app_state
    sparse_index = state.sparse_index

    try:
        docs = sparse_index.get_chunks_by_source(source_name)
        metadatas: list[dict[str, Any]] = [doc.metadata for doc in docs]
        palette = [
            {"bg": "#e0e7ff", "text": "#3730a3", "border": "#818cf8"}, # Indigo
            {"bg": "#dcfce7", "text": "#166534", "border": "#4ade80"}, # Green
            {"bg": "#ffedd5", "text": "#9a3412", "border": "#fb923c"}, # Orange
            {"bg": "#f3e8ff", "text": "#5b21b6", "border": "#c084fc"}, # Purple
            {"bg": "#cffafe", "text": "#155e75", "border": "#22d3ee"}, # Cyan
            {"bg": "#fce7f3", "text": "#86198f", "border": "#f472b6"}, # Fuchsia
            {"bg": "#fef3c7", "text": "#c2410c", "border": "#fbbf24"}, # Amber
            {"bg": "#ecfdf5", "text": "#065f46", "border": "#34d399"}  # Emerald
        ]

        sections: dict[str, list[dict]] = {}
        for meta in metadatas[:150]:  # Limit to 150 to keep graph sane
            sec = meta.get("section", "Unknown")
            if sec not in sections:
                sections[sec] = []
            sections[sec].append(meta)

        nodes = []
        edges = []

        cluster_x = 350
        chunk_x = 700
        current_y = 50

        for c_idx, (section, sec_metas) in enumerate(sections.items()):
            cluster_id = f"cluster_{c_idx}"
            color = palette[c_idx % len(palette)]
            
            start_y = current_y
            max_y_in_cluster = start_y
            
            # Add Chunks first to calculate vertical span
            for idx, meta in enumerate(sec_metas):
                chunk_id = meta.get("chunk_id", f"chunk_{c_idx}_{idx}")
                
                # Distribute horizontally in a 3-column grid to reduce vertical scroll
                cols = 3
                col = idx % cols
                row = idx // cols
                
                node_x = chunk_x + (col * 220)
                node_y = start_y + (row * 70)
                max_y_in_cluster = max(max_y_in_cluster, node_y)
                
                nodes.append({
                    "id": chunk_id,
                    "data": {"label": f"Chunk {idx}\nType: {meta.get('chunk_type', 'text')}"},
                    "position": {"x": node_x, "y": node_y},
                    "style": {
                        "background": "#ffffff", 
                        "color": color["text"],
                        "fontSize": "10px", 
                        "padding": "4px", 
                        "borderRadius": "4px",
                        "border": f"2px solid {color['border']}",
                        "width": 180
                    }
                })

                # Edge from cluster to chunk
                edges.append({
                    "id": f"e_cl_{cluster_id}_{chunk_id}",
                    "source": cluster_id,
                    "target": chunk_id,
                    "type": "smoothstep",
                    "style": {"stroke": color["border"]}
                })
                
                # Sequential coherence edge inside cluster
                if idx > 0:
                    prev_chunk = sec_metas[idx - 1].get("chunk_id", f"chunk_{c_idx}_{idx - 1}")
                    edges.append({
                        "id": f"e_seq_{prev_chunk}_{chunk_id}",
                        "source": prev_chunk,
                        "target": chunk_id,
                        "type": "straight",
                        "style": {"stroke": color["border"], "strokeDasharray": "5,5"}
                    })

            current_y = max_y_in_cluster + 80 # space after the last row of chunks

            # Add Cluster Node aligned to the vertical center of its chunks
            cluster_y = start_y + ((max_y_in_cluster - start_y) / 2)
            
            nodes.append({
                "id": cluster_id,
                "data": {"label": f"Section: {section[:30]}...", "type": "cluster"},
                "position": {"x": cluster_x, "y": cluster_y},
                "type": "default",
                "style": {
                    "background": color["bg"], 
                    "color": color["text"],
                    "border": f"2px solid {color['border']}", 
                    "borderRadius": "8px",
                    "padding": "10px",
                    "width": 220
                }
            })
            
            # Edge from root to cluster
            edges.append({
                "id": f"e_root_{cluster_id}", 
                "source": "root", 
                "target": cluster_id, 
                "animated": True, 
                "type": "smoothstep",
                "style": {"stroke": "#94a3b8", "strokeWidth": 2}
            })

            current_y += 50 # padding between sections

        # Add Root node centered vertically
        nodes.append({
            "id": "root", 
            "data": {"label": source_name, "type": "document"}, 
            "position": {"x": 50, "y": (current_y - 50) / 2},
            "type": "default",
            "style": {
                "background": "#f8fafc", 
                "color": "#0f172a",
                "border": "3px solid #cbd5e1", 
                "borderRadius": "8px", 
                "padding": "16px",
                "fontWeight": "bold"
            }
        })

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        logger.error("Failed to generate graph: %s", e)
        return JSONResponse({"nodes": [], "edges": []})

@router.get("/{source_name}/section/{section_id}")
async def get_section_chunks(
    source_name: str, section_id: str, request: Request
) -> JSONResponse:
    """
    Return the actual text chunks belonging to a specific section node.
    Used by the frontend tree browser when a user clicks into a section.
    """
    state = request.app.state.app_state
    sparse_index = state.sparse_index

    try:
        all_docs = sparse_index.get_chunks_by_source(source_name)
        
        chunks = [
            {"text": doc.page_content, "meta": doc.metadata}
            for doc in all_docs
            if doc.metadata.get("section_id") == section_id
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
