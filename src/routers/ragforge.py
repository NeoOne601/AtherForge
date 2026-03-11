import asyncio
import structlog
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from src.utils import safe_create_task

router = APIRouter(prefix="/api/v1/ragforge", tags=["RAGForge"])
logger = structlog.get_logger("aetherforge.ragforge")

@router.get("/documents")
async def list_rag_documents(request: Request, limit: int = 50, offset: int = 0) -> JSONResponse:
    state = request.app.state.app_state
    if not state.vector_store:
        return JSONResponse({"documents": [], "total": 0})

    try:
        col = state.vector_store._collection
        # In a real scenario, we would use limit/offset in the query, 
        # but Chroma's .get() has limit/offset support.
        res = await asyncio.to_thread(col.get, include=["metadatas"])
        metadatas = res.get("metadatas", [])
        
        doc_map = {}
        for m in metadatas:
            if not m: continue
            src = m.get("source")
            if not src: continue
            doc_map[src] = doc_map.get(src, 0) + 1
            
        all_docs = []
        for src, count in doc_map.items():
            all_docs.append({
                "name": src,
                "status": "Ready",
                "tokens": f"~{count} chunks"
            })
            
        # Apply manual pagination for the grouped document list
        paginated_docs = all_docs[offset : offset + limit]
            
        return JSONResponse({
            "documents": paginated_docs,
            "total": len(all_docs),
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        logger.error("Failed to list ragforge documents: %s", e)
        return JSONResponse({"documents": [], "total": 0})

@router.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    state = request.app.state.app_state
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_path = state.settings.data_dir / "uploads" / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        from src.modules.ragforge_indexer import index_document
        result = await asyncio.to_thread(
            index_document, file_path, state.vector_store, state.sparse_index
        )
        return JSONResponse({"status": "Success", "filename": file.filename, "result": result})
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vlm-options")
async def get_vlm_options(request: Request):
    """List available Vision Language Models for RAG enrichment."""
    import platform
    from src.modules.ragforge.vlm_provider import list_providers
    
    providers = list_providers()
    is_mac = platform.system() == "Darwin"
    
    options = []
    default_vlm = "smolvlm-256m"
    
    # Logic for hardware rating and default selection
    for p in providers:
        hw_rating = "optimal"
        if p.id == "apple-vlm" and not is_mac:
            continue # Only show Apple VLM on Mac
        
        if p.id == "ollama-qwen3.5-9b":
            hw_rating = "warning" # High RAM requirement
            
        options.append({
            "id": p.id,
            "name": p.name,
            "hardware_rating": hw_rating,
            "tier": p.tier
        })
        
        # Default to apple-vlm on Mac if available
        if is_mac and p.id == "apple-vlm":
            default_vlm = "apple-vlm"
            
    return {"options": options, "selected": default_vlm}

@router.post("/vlm-select")
async def select_vlm(payload: dict, request: Request):
    """Select the VLM to use for image/pdf enrichment."""
    vlm_id = payload.get("vlm_id")
    logger.info("VLM selection updated", vlm_id=vlm_id)
    return {"status": "ok", "vlm_id": vlm_id}

