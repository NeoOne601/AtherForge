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
        hw_rating = "optimal"
        if provider.id == "apple-vlm" and not is_mac:
            continue
        if provider.id == "ollama-qwen3.5-9b":
            hw_rating = "warning"

        options.append(
            {
                "id": provider.id,
                "name": provider.name,
                "hardware_rating": hw_rating,
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
