import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.settings_store import save_partial_settings

router = APIRouter(prefix="/api/v1/ragforge", tags=["RAGForge"])
logger = structlog.get_logger("aetherforge.ragforge")


class DocumentSelectionRequest(BaseModel):
    selected: bool


@router.get("/documents")
async def list_rag_documents(request: Request, limit: int = 100, offset: int = 0) -> JSONResponse:
    state = request.app.state.app_state
    records = state.document_registry.list_documents(limit=limit, offset=offset)
    documents = []
    for record in records:
        payload = record.to_dict()
        payload["name"] = record.source
        payload["status"] = record.ingest_status
        payload["tokens"] = f"~{record.chunk_count} chunks"
        documents.append(payload)

    return JSONResponse(
        {
            "documents": documents,
            "total": state.document_registry.count_documents(),
            "limit": limit,
            "offset": offset,
        }
    )


@router.patch("/documents/{document_id}")
async def update_document_selection(
    document_id: str,
    payload: DocumentSelectionRequest,
    request: Request,
) -> JSONResponse:
    state = request.app.state.app_state
    updated = state.document_registry.update_document(document_id, selected=payload.selected)
    if updated is None:
        raise HTTPException(status_code=404, detail="Document not found")
    body = updated.to_dict()
    body["name"] = updated.source
    body["status"] = updated.ingest_status
    body["tokens"] = f"~{updated.chunk_count} chunks"
    return JSONResponse(body)


@router.post("/documents/{document_id}/retry")
async def retry_document_ingest(document_id: str, request: Request) -> JSONResponse:
    """Manually retry a failed or partial document ingestion."""
    state = request.app.state.app_state
    record = state.document_registry.get_by_id(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = state.settings.uploads_dir / record.source
    if not file_path.exists():
        # Fallback to LiveFolder if not in uploads
        file_path = state.settings.data_dir / "LiveFolder" / record.source

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Source file '{record.source}' not found on disk.")

    result = await state.document_intelligence.ingest_path(file_path, force=True)
    return JSONResponse(result)


@router.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    state = request.app.state.app_state
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_path = state.settings.uploads_dir / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        with open(file_path, "wb") as handle:
            handle.write(content)

        result = await state.document_intelligence.ingest_path(file_path)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Upload failed", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/vlm-options")
async def get_vlm_options(request: Request):
    """List available Vision Language Models for RAG enrichment."""
    import platform

    from src.modules.ragforge.vlm_provider import list_providers

    state = request.app.state.app_state
    providers = list_providers()
    is_mac = platform.system() == "Darwin"

    options = []
    default_vlm = getattr(state, "selected_vlm_id", "smolvlm-256m")

    for provider in providers:
        if provider.id == "apple-vlm" and not is_mac:
            continue
            
        is_safe, msg = provider.is_hardware_safe()
        hw_rating = "optimal" if is_safe else "warning"

        options.append(
            {
                "id": provider.id,
                "name": provider.name,
                "hardware_rating": hw_rating,
                "hardware_message": msg,
                "tier": provider.tier,
            }
        )
        if is_mac and provider.id == "apple-vlm" and default_vlm == "smolvlm-256m":
            default_vlm = "apple-vlm"

    return {"options": options, "selected": default_vlm}


@router.post("/vlm-select")
async def select_vlm(payload: dict, request: Request):
    """Select the VLM to use for image/pdf enrichment."""
    vlm_id = str(payload.get("vlm_id", "smolvlm-256m"))
    state = request.app.state.app_state
    state.selected_vlm_id = vlm_id
    save_partial_settings({"SELECTED_VLM_ID": vlm_id})
    logger.info("VLM selection updated", vlm_id=vlm_id)
    return {"status": "ok", "vlm_id": vlm_id, "selected": vlm_id}


@router.post("/documents/{document_id}/enrich-images")
async def enrich_document_images(document_id: str, request: Request) -> JSONResponse:
    """
    Trigger VLM enrichment for only the pending image pages of a document.
    This is the backend for the '🖼 Enrich Images' button shown for 'partial'
    and 'ocr_pending' documents where text was extracted but images were not.
    """
    import fitz
    from src.modules.ragforge_indexer import _analyze_pdf
    from src.utils import safe_create_task

    state = request.app.state.app_state
    record = state.document_registry.get_by_id(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    # Resolve file path
    file_path = state.settings.uploads_dir / record.source
    if not file_path.exists():
        file_path = state.settings.data_dir / "LiveFolder" / record.source
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Source file '{record.source}' not found on disk.")

    # Re-analyze to find which pages have images (not covered by text extraction)
    try:
        analysis = _analyze_pdf(file_path)
        image_pages = analysis.get("image_pages", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF analysis failed: {e}")

    if not image_pages:
        return JSONResponse({
            "status": "no_images",
            "message": "No image pages found in this document.",
            "document_id": document_id,
        })

    # Check VLM is reachable before queuing
    vlm_id = str(getattr(state, "selected_vlm_id", None) or "smolvlm-256m")

    # Update status to indicate VLM enrichment is pending
    state.document_registry.update_document(
        document_id,
        ingest_status="ocr_running",
        last_error=None,
        image_pages_pending=len(image_pages),
    )

    # Queue the VLM enrichment as a background task
    safe_create_task(
        state.document_intelligence._run_vlm_enrichment(file_path, document_id, image_pages),
        name=f"vlm_enrich_manual_{record.source}",
    )

    logger.info(
        "Manual VLM enrichment queued",
        source=record.source,
        document_id=document_id,
        image_pages=len(image_pages),
        vlm_id=vlm_id,
    )
    return JSONResponse({
        "status": "queued",
        "message": f"VLM enrichment started for {len(image_pages)} image page(s) using {vlm_id}.",
        "document_id": document_id,
        "image_pages": len(image_pages),
        "vlm_id": vlm_id,
    })


@router.post("/documents/{document_id}/force-reindex")
async def force_reindex_document(document_id: str, request: Request) -> JSONResponse:
    """
    Force a full re-index of a document, bypassing the idempotency guard.
    Use this when you've updated the file on disk and want to pick up new content.
    """
    state = request.app.state.app_state
    record = state.document_registry.get_by_id(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = state.settings.uploads_dir / record.source
    if not file_path.exists():
        file_path = state.settings.data_dir / "LiveFolder" / record.source
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Source file '{record.source}' not found on disk.")

    logger.info("Force re-index triggered", source=record.source, document_id=document_id)
    # Update document status to signal re-index is in progress
    state.document_registry.update_document(
        document_id, ingest_status="extracting_text", last_error=None, chunk_count=0
    )
    result = await state.document_intelligence.ingest_path(file_path, force=True)
    return JSONResponse(result)


@router.get("/live-folder")
async def list_live_folder(request: Request) -> JSONResponse:
    """
    Returns all files currently in the LiveFolder with their state.
    Powers the WatchTower 'Live Ingestion Feed' panel showing real-time ingestion status.
    """
    import os
    from pathlib import Path

    state = request.app.state.app_state
    live_folder: Path = state.settings.data_dir / "LiveFolder"
    
    if not live_folder.exists():
        return JSONResponse({"files": [], "total": 0, "folder": str(live_folder)})

    # Get all files in LiveFolder
    file_entries = []
    for fp in sorted(live_folder.iterdir()):
        if not fp.is_file():
            continue
        try:
            stat = fp.stat()
            size_kb = round(stat.st_size / 1024, 1)
            # Get indexed state from document registry
            record = state.document_registry.get_by_source(fp.name)
            entry = {
                "name": fp.name,
                "size_kb": size_kb,
                "modified": stat.st_mtime,
                "extension": fp.suffix.lower(),
                "status": record.ingest_status if record else "not_indexed",
                "chunk_count": record.chunk_count if record else 0,
                "image_pages_pending": record.image_pages_pending if record else 0,
                "document_id": record.document_id if record else None,
                "last_error": record.last_error if record else None,
            }
            file_entries.append(entry)
        except Exception:
            pass

    return JSONResponse({
        "files": file_entries,
        "total": len(file_entries),
        "folder": str(live_folder),
    })


@router.get("/ingestion-progress")
async def get_ingestion_progress() -> JSONResponse:
    """Real-time ingestion progress for all currently processing documents.

    Returns per-document metrics:
      - current_page / total_pages — how far Docling has processed
      - chunks_so_far — chunks extracted so far
      - batch_size — adaptive batch size being used
      - last_batch_seconds — how long the most recent batch took
      - eta_seconds — estimated time remaining based on running average
      - status — 'extracting_text' or 'indexing_complete'
    """
    from src.modules.ragforge_indexer import _ingestion_progress

    # Strip batch_times list (internal metric, too verbose for API)
    progress_data = {}
    for filename, info in _ingestion_progress.items():
        progress_data[filename] = {
            "current_page": info.get("current_page", 0),
            "total_pages": info.get("total_pages", 0),
            "chunks_so_far": info.get("chunks_so_far", 0),
            "batch_size": info.get("batch_size", 10),
            "last_batch_seconds": info.get("last_batch_seconds", 0),
            "eta_seconds": info.get("eta_seconds", 0),
            "status": info.get("status", "unknown"),
            "percent": round(
                100 * info.get("current_page", 0) / max(info.get("total_pages", 1), 1),
                1,
            ),
        }

    return JSONResponse({"progress": progress_data})


class PageHitRequest(BaseModel):
    source: str
    page: int


@router.post("/record-hit")
async def record_page_hit(payload: PageHitRequest, request: Request) -> JSONResponse:
    """
    Record that a user has 'accessed' or 'viewed' a specific page.
    This increases the priority of this page in the VLM enrichment queue.
    """
    state = request.app.state.app_state
    state.document_registry.record_page_hit(payload.source, payload.page)
    logger.info("Page hit recorded (Priority +1)", source=payload.source, page=payload.page)
    return JSONResponse({"status": "ok", "source": payload.source, "page": payload.page})
