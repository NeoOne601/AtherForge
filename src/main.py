# AetherForge v1.0 — src/main.py
# ─────────────────────────────────────────────────────────────────
# FastAPI application entry point. This is the entire backend for
# AetherForge — a single ASGI server on localhost:8765 that:
#
#   1. Manages BitNet LLM lifecycle (load once, reuse always)
#   2. Exposes REST + WebSocket endpoints for the Tauri frontend
#   3. Runs the LangGraph meta-agent on each chat request
#   4. Writes every interaction to the encrypted replay buffer
#   5. Schedules nightly OPLoRA fine-tuning jobs
#
# Design decision: Single-process, embedded services.
# We intentionally avoid microservices for local deployment. One
# uvicorn process with APScheduler + background tasks keeps the
# app within the <150 ms cold-start target and <4 GB RAM budget.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

# ── Model cache redirect — MUST be set before any HuggingFace/torch import ──
# Redirects ALL model downloads to the external drive so the iMac internal
# SSD is not consumed by large model files. Reads HF_HOME from environment
# (set in .env / run_dev.sh) with a fallback to the known external volume.
import os as _os
_EXTERNAL_AI_DRIVE = _os.environ.get(
    "HF_HOME",
    "/Volumes/Apple/AI Model/hf_cache"
)
_hf_hub = f"{_EXTERNAL_AI_DRIVE}/hub"
_os.environ.setdefault("HF_HOME",                        _EXTERNAL_AI_DRIVE)
_os.environ.setdefault("HF_HUB_CACHE",                   _hf_hub)
_os.environ.setdefault("TRANSFORMERS_CACHE",              _hf_hub)
_os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",      f"{_EXTERNAL_AI_DRIVE}/sentence_transformers")
_os.environ.setdefault("TORCH_HOME",                     f"{_EXTERNAL_AI_DRIVE}/torch")
_os.environ.setdefault("DOCLING_CACHE_DIR",              f"{_EXTERNAL_AI_DRIVE}/docling")
# ─────────────────────────────────────────────────────────────────

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import psutil
import typer
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, UploadFile, File, Request, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AetherForgeSettings, get_settings
from src.guardrails.silicon_colosseum import SiliconColosseum
import uuid
from src.learning.replay_buffer import ReplayBuffer
from src.meta_agent import MetaAgent, MetaAgentInput
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("aetherforge.main")

# ── Shared application state ──────────────────────────────────────
# We store heavyweight singletons in AppState rather than globals
# so they're accessible via request.app.state in any endpoint.
class AppState:
    settings: AetherForgeSettings
    meta_agent: MetaAgent
    replay_buffer: ReplayBuffer
    colosseum: SiliconColosseum
    scheduler: AsyncIOScheduler
    vector_store: Any
    sparse_index: Any          # SparseIndex singleton — shared to prevent orphan DB writes
    export_engine: Any         # ExportEngine — MD + PDF export
    startup_ms: float
    selected_vlm_id: str = "smolvlm-256m"  # Default to Lite tier for 8GB safety

    # StreamSync additions
    streamsync_rss_feeds: list[str] = []
    directory_watcher: Any = None

    # Sync
    sync_manager: Any = None


# ── Lifespan (replaces @app.on_event) ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: initialise all heavyweight components once.
    Shutdown: flush buffers, cancel scheduler, close connections.

    Using asynccontextmanager lifespan is the FastAPI 0.93+ best
    practice — avoids double-init bugs from @on_event decorators.
    """
    t0 = time.perf_counter()
    state = AppState()
    app.state.app_state = state

    # ── 1. Load settings ─────────────────────────────────────────
    settings = get_settings()
    state.settings = settings
    logger.info(
        "AetherForge v1.0 starting | env=%s | port=%d",
        settings.aetherforge_env,
        settings.aetherforge_port,
    )

    # ── 2. Initialise encrypted replay buffer ─────────────────────
    state.replay_buffer = ReplayBuffer(settings)
    await state.replay_buffer.initialize()
    logger.info("Replay buffer ready at %s", settings.replay_buffer_path)

    # ── 2.5 Initialise local Vector Store (ChromaDB + BGE-M3) ─────
    # all-MiniLM-L6-v2 — lightweight embedding model for 8GB edge devices:
    #   - 22M params, 80MB model file (vs BGE-M3's 2.27GB)
    #   - 384 dimensions, 512 token limit
    #   - Embeds at ~3000 sentences/sec on CPU (vs BGE-M3's ~10 sentences/sec)
    #   - Combined with FTS5/BM25 hybrid search, keyword matching compensates
    #     for the smaller embedding space — best speed/quality for edge devices.
    logger.info("Initializing ChromaDB with all-MiniLM-L6-v2 embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    state.vector_store = Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
    )
    logger.info("ChromaDB active at %s | model=all-MiniLM-L6-v2 | dims=384",
                settings.chroma_path)

    # ── 2.6 Initialise shared SparseIndex (FTS5/BM25) ────────────────
    # Singleton shared across upload + VLM background paths to ensure
    # all chunks (text AND visual) go to the same SQLite database.
    from src.modules.ragforge.sparse_index import SparseIndex
    state.sparse_index = SparseIndex(
        db_path=settings.data_dir / "sparse_index.db"
    )
    logger.info("SparseIndex (FTS5/BM25) active at %s", settings.data_dir / "sparse_index.db")

    # ── 2.7 Initialise SessionStore + ExportEngine ────────────────
    from src.modules.session_store import SessionStore
    from src.modules.export_engine import ExportEngine
    state.session_store = SessionStore(
        db_path=settings.data_dir / "sessions.db"
    )
    state.export_engine = ExportEngine(state.session_store)
    logger.info("SessionStore active at %s", settings.data_dir / "sessions.db")

    # ── 3. Initialise Silicon Colosseum (OPA + FSM) ───────────────
    state.colosseum = SiliconColosseum(settings)
    await state.colosseum.initialize()
    logger.info("Silicon Colosseum guardrails active | mode=%s", settings.opa_mode)

    # ── 4. Initialise LangGraph Meta-Agent (loads BitNet model) ───
    state.meta_agent = MetaAgent(settings, state.colosseum, state.vector_store, state.replay_buffer)
    await state.meta_agent.initialize()
    logger.info("Meta-agent ready | modules=5 | model=%s", settings.bitnet_model_path.name)

    # ── 5. Start nightly OPLoRA scheduler ─────────────────────────
    state.scheduler = AsyncIOScheduler(timezone="UTC")
    state.scheduler.add_job(
        _nightly_oplora_job,
        trigger=CronTrigger(
            hour=settings.oploра_nightly_hour,
            minute=settings.oploра_nightly_minute,
        ),
        args=[app],
        id="nightly_oplora",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    state.scheduler.start()
    logger.info(
        "Nightly OPLoRA scheduled at %02d:%02d local time",
        settings.oploра_nightly_hour,
        settings.oploра_nightly_minute,
    )

    state.startup_ms = (time.perf_counter() - t0) * 1000
    logger.info("AetherForge startup complete in %.1f ms", state.startup_ms)

    # ── 6. Start WatchTower live metric poller (psutil every 2s) ──────
    _poller_task = asyncio.create_task(_watchtower_poller())
    logger.info("WatchTower psutil poller started")

    # ── 7. Start StreamSync RSS & Directory Watchers ──────────────────
    from src.modules.streamsync.rss_feeder import rss_poller_task
    from src.modules.streamsync.directory_watcher import StreamSyncDirectoryWatcher
    
    # Load RSS feeds from settings
    rss_state_file = settings.data_dir / "streamsync_rss_feeds.json"
    if rss_state_file.exists():
        try:
            with open(rss_state_file, "r") as f:
                state.streamsync_rss_feeds = json.load(f).get("feeds", [])
        except Exception as e:
            logger.error("Failed to load StreamSync RSS feeds: %s", e)
    
    _rss_poller_task = asyncio.create_task(rss_poller_task(app))
    
    live_folder = settings.data_dir / "LiveFolder"
    loop = asyncio.get_running_loop()
    state.directory_watcher = StreamSyncDirectoryWatcher(live_folder, loop)
    state.directory_watcher.start()

    # ── 8. Start Multi-Node Synchronizer ──────────────────────────────
    from src.modules.sync.event_log import EventLog
    from src.modules.sync.sync_manager import SyncManager
    node_id_file = settings.data_dir / "node_id.txt"
    if not node_id_file.exists():
        node_id_file.write_text(str(uuid.uuid4()))
    node_id = node_id_file.read_text().strip()

    event_log = EventLog(settings.data_dir / "sync_events.db")
    state.sync_manager = SyncManager(node_id, settings.aetherforge_port, event_log)
    await state.sync_manager.start()

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────
    logger.info("AetherForge shutting down...")
    _poller_task.cancel()
    _rss_poller_task.cancel()
    if state.directory_watcher:
        state.directory_watcher.stop()
    if state.sync_manager:
        await state.sync_manager.stop()
    state.scheduler.shutdown(wait=False)
    await state.replay_buffer.flush()
    logger.info("Shutdown complete.")


# ── FastAPI App ───────────────────────────────────────────────────
app = FastAPI(
    title="AetherForge API",
    description="Local AI OS — glass-box perpetual-learning backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS: only allow Tauri's protocol-asset origin and localhost Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "tauri://localhost",
        "https://tauri.localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────
def get_state(app_: Any) -> AppState:
    return app_.state.app_state  # type: ignore[no-any-return]


# ── WatchTower Live Metric Poller ─────────────────────────────
async def _watchtower_poller() -> None:
    """
    Background task: polls psutil every 2 seconds and feeds real CPU,
    memory, and network throughput into WatchTower's metric windows.
    This is what populates the telemetry dashboard automatically without
    requiring external webhooks.
    """
    from src.modules.watchtower.graph import ingest_metric
    net_prev = psutil.net_io_counters()
    prev_time = time.perf_counter()
    try:
        while True:
            await asyncio.sleep(2)
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                mem_pct = psutil.virtual_memory().percent
                # Network throughput in MB/s
                net_curr = psutil.net_io_counters()
                curr_time = time.perf_counter()
                elapsed = max(curr_time - prev_time, 0.001)
                net_mb = (net_curr.bytes_sent + net_curr.bytes_recv -
                          net_prev.bytes_sent - net_prev.bytes_recv) / (elapsed * 1024 * 1024)
                net_prev = net_curr
                prev_time = curr_time

                ingest_metric("cpu", cpu_pct)
                ingest_metric("mem", mem_pct)
                ingest_metric("net", round(net_mb, 2))
            except Exception as e:
                logger.warning("WatchTower poller tick failed: %s", e)
    except asyncio.CancelledError:
        logger.info("WatchTower poller stopped")


# ─────────────────────────────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/streamsync/rss", tags=["StreamSync"])
async def get_rss_feeds() -> JSONResponse:
    state: AppState = get_state(app)
    return JSONResponse({"feeds": state.streamsync_rss_feeds})

class RSSFeedRequest(BaseModel):
    url: str

@app.post("/api/v1/streamsync/rss", tags=["StreamSync"])
async def add_rss_feed(req: RSSFeedRequest) -> JSONResponse:
    state: AppState = get_state(app)
    if req.url not in state.streamsync_rss_feeds:
        state.streamsync_rss_feeds.append(req.url)
        rss_state_file = state.settings.data_dir / "streamsync_rss_feeds.json"
        with open(rss_state_file, "w") as f:
            json.dump({"feeds": state.streamsync_rss_feeds}, f)
        from src.modules.streamsync.graph import emit_event
        emit_event("rss_feed_added", payload={"url": req.url}, source="System")
    return JSONResponse({"status": "success", "feeds": state.streamsync_rss_feeds})

@app.delete("/api/v1/streamsync/rss", tags=["StreamSync"])
async def remove_rss_feed(req: RSSFeedRequest) -> JSONResponse:
    state: AppState = get_state(app)
    if req.url in state.streamsync_rss_feeds:
        state.streamsync_rss_feeds.remove(req.url)
        rss_state_file = state.settings.data_dir / "streamsync_rss_feeds.json"
        with open(rss_state_file, "w") as f:
            json.dump({"feeds": state.streamsync_rss_feeds}, f)
        from src.modules.streamsync.graph import emit_event
        emit_event("rss_feed_removed", payload={"url": req.url}, source="System")
    return JSONResponse({"status": "success", "feeds": state.streamsync_rss_feeds})

@app.get("/api/v1/health", tags=["System"])
async def health() -> JSONResponse:
    """
    Lightweight health check used by run_dev.sh and Tauri startup
    probe. Returns 200 when the backend is ready to serve requests.
    """
    state: AppState = get_state(app)
    return JSONResponse({
        "status": "ok",
        "env": state.settings.aetherforge_env,
        "startup_ms": round(state.startup_ms, 1),
        "model": state.settings.bitnet_model_path.name,
    })


@app.get("/api/v1/status", tags=["System"])
async def system_status() -> JSONResponse:
    """
    Detailed system status for the frontend dashboard.
    Includes battery %, CPU load, memory usage, and module states.
    """
    battery = psutil.sensors_battery()
    return JSONResponse({
        "battery_pct": battery.percent if battery else None,
        "battery_plugged": battery.power_plugged if battery else None,
        "cpu_pct": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "modules": ["ragforge", "localbuddy", "watchtower", "streamsync", "tunelab"],
    })

@app.get("/api/v1/ragforge/documents", tags=["RAGForge"])
async def list_rag_documents() -> JSONResponse:
    """Returns a list of all distinct documents currently indexed in ChromaDB."""
    state: AppState = get_state(app)
    if not state.vector_store:
        return JSONResponse({"documents": []})

    try:
        col = state.vector_store._collection
        res = await asyncio.to_thread(col.get, include=["metadatas"])
        metadatas = res.get("metadatas", [])
        
        doc_map = {}
        for m in metadatas:
            if not m: continue
            src = m.get("source")
            if not src: continue
            doc_map[src] = doc_map.get(src, 0) + 1
            
        docs = []
        for src, count in doc_map.items():
            docs.append({
                "name": src,
                "status": "Ready",
                "tokens": f"~{count} chunks",
                "active": True
            })
            
        return JSONResponse({"documents": docs})
    except Exception as e:
        logger.error("Failed to list ragforge documents: %s", e)
        return JSONResponse({"documents": []})
@app.post("/api/v1/ragforge/upload", tags=["RAGForge"])
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """
    Ingest a document (PDF, MD, CSV, TXT) into the local vector DB.
    
    Pipeline:
      1. Save file to disk
      2. Docling text extraction + embedding (fast, <1 min) → returns immediately  
      3. VLM figure extraction (background async) → enriches index while user queries
    """
    state: AppState = get_state(app)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_path = state.settings.data_dir / "uploads" / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save file to disk
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # ── Fast path: text extraction + embedding (<1 min) ──────
        from src.modules.ragforge_indexer import index_document
        result = await asyncio.to_thread(
            index_document, file_path, state.vector_store, state.sparse_index
        )

        chunks_added = result.get("chunks_added", 0) if isinstance(result, dict) else int(result)
        parser = result.get("parser", "unknown") if isinstance(result, dict) else "legacy"
        breakdown = result.get("chunk_breakdown", {}) if isinstance(result, dict) else {}
        image_pages = result.get("image_pages", []) if isinstance(result, dict) else []

        logger.info(
            "Indexed '%s' — %d chunks via %s | breakdown=%s",
            file.filename, chunks_added, parser, breakdown,
        )

        # ── Async VLM: enrich figures in background ──────────────
        if image_pages and file_path.suffix.lower() == ".pdf" and getattr(state, "selected_vlm_id", None):
            asyncio.create_task(
                _async_vlm_enrich(
                    file_path, image_pages, state.vector_store,
                    state.selected_vlm_id, state.sparse_index
                )
            )
            logger.info("VLM background task spawned for %d image pages", len(image_pages))

        return JSONResponse({
            "status": "success",
            "chunks_indexed": chunks_added,
            "parser": parser,
            "chunk_breakdown": breakdown,
            "file": file.filename,
            "vlm_background": len(image_pages) > 0 and bool(state.selected_vlm_id),
        })
    except Exception as e:
        logger.exception("Failed to process RAG document %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e)) from e



# ════════════════════════════════════════════════════════════════
# Session Management API
# ════════════════════════════════════════════════════════════════

@app.get("/api/v1/sessions", tags=["Sessions"])
async def list_sessions(
    request: Request,
    module: str | None = None,
) -> JSONResponse:
    """List all chat sessions, optionally filtered by module."""
    state: AppState = request.app.state.app_state
    sessions = state.session_store.list_sessions(module=module)
    return JSONResponse([
        {
            "id": s.id,
            "module": s.module,
            "title": s.title,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "message_count": s.message_count,
        }
        for s in sessions
    ])


@app.delete("/api/v1/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str, request: Request) -> JSONResponse:
    """Delete a session and all its messages."""
    state: AppState = request.app.state.app_state
    state.session_store.delete_session(session_id)
    # Also evict from in-process MetaAgent memory cache
    state.meta_agent._session_memories.pop(session_id, None)
    return JSONResponse({"status": "deleted", "session_id": session_id})


class RenameSessionRequest(BaseModel):
    title: str


@app.patch("/api/v1/sessions/{session_id}", tags=["Sessions"])
async def rename_session(
    session_id: str,
    req: RenameSessionRequest,
    request: Request,
) -> JSONResponse:
    """Rename a session."""
    state: AppState = request.app.state.app_state
    state.session_store.rename_session(session_id, req.title)
    return JSONResponse({"status": "renamed", "session_id": session_id, "title": req.title})


@app.get("/api/v1/sessions/{session_id}/messages", tags=["Sessions"])
async def get_session_messages(session_id: str, request: Request) -> JSONResponse:
    """Return all messages for a session (for UI history reload)."""
    state: AppState = request.app.state.app_state
    messages = state.session_store.get_messages(session_id)
    return JSONResponse([
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "ts": m.ts,
            "metadata": m.metadata,
        }
        for m in messages
    ])


# ════════════════════════════════════════════════════════════════
# Export API  (MD + PDF)
# ════════════════════════════════════════════════════════════════

from fastapi.responses import Response as FastAPIResponse


@app.get("/api/v1/sessions/{session_id}/export", tags=["Export"])
async def export_session(
    session_id: str,
    request: Request,
    format: str = "md",               # "md" | "pdf"
    message_id: str | None = None,    # If set, export only this message
) -> FastAPIResponse:
    """
    Export a full session (or a single message) as Markdown or PDF.

    Query params:
        format=md   → returns a .md file download
        format=pdf  → returns a .pdf file download
        message_id  → if provided, export only that single AI response
    """
    state: AppState = request.app.state.app_state
    engine = state.export_engine
    safe_id = session_id[:8]

    try:
        if format == "pdf":
            if message_id:
                content = await asyncio.to_thread(
                    engine.message_to_pdf, session_id, message_id
                )
                filename = f"aetherforge_response_{safe_id}.pdf"
            else:
                content = await asyncio.to_thread(
                    engine.session_to_pdf, session_id
                )
                filename = f"aetherforge_session_{safe_id}.pdf"
            return FastAPIResponse(
                content=content,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            # Default: markdown
            if message_id:
                content = engine.message_to_markdown(session_id, message_id)
                filename = f"aetherforge_response_{safe_id}.md"
            else:
                content = engine.session_to_markdown(session_id)
                filename = f"aetherforge_session_{safe_id}.md"
            return FastAPIResponse(
                content=content.encode("utf-8"),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
    except Exception as e:
        logger.error("Export failed for session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}") from e


@app.get("/api/v1/ragforge/vlm-options", tags=["RAGForge"])
async def get_vlm_options() -> JSONResponse:
    """Returns available VLM providers with hardware recommendations."""
    from src.modules.ragforge.vlm_provider import list_providers
    options = []
    
    # Get total system RAM to suggest safety
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    avail_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    for p in list_providers():  # uses singleton registry — no torch re-init
        is_safe = p.required_ram_gb < avail_ram_gb * 0.75  # leave 25% headroom
        options.append({
            "id": p.id,
            "name": p.name,
            "tier": p.tier,
            "required_ram_gb": p.required_ram_gb,
            "hardware_rating": "safe" if is_safe else "warning",
        })
        
    return JSONResponse({"options": options})

class VLMSelectRequest(BaseModel):
    vlm_id: str

@app.post("/api/v1/ragforge/vlm-select", tags=["RAGForge"])
async def select_vlm(req: VLMSelectRequest) -> JSONResponse:
    """Selects the active VLM and triggers background compilation/download."""
    from src.modules.ragforge.vlm_provider import get_vlm_provider
    state: AppState = get_state(app)
    
    provider = get_vlm_provider(req.vlm_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"VLM ID '{req.vlm_id}' not found")
        
    state.selected_vlm_id = req.vlm_id
    logger.info("VLM Selection updated: %s", req.vlm_id)
    
    # Optionally trigger background load to pre-warm
    asyncio.create_task(provider.load_model())
    
    return JSONResponse({"status": "success", "selected": req.vlm_id})


class ChatModelSelectRequest(BaseModel):
    model_id: str

@app.get("/api/v1/chat-models", tags=["System"])
async def get_chat_models() -> JSONResponse:
    """Returns available optimized chat models for the 8GB RAM constraint."""
    state: AppState = get_state(app)
    
    models = [
        {"id": "bitnet-2b", "name": "BitNet b1.58 2B (Default)"},
        {"id": "gemma2-2b", "name": "Gemma 2 2B (INT4)"},
        {"id": "llama3.2-3b", "name": "Llama 3.2 3B (INT4)"}
    ]
    
    # Check if the active model filename contains the id keywords to set current selection
    active_name = state.settings.bitnet_model_path.name.lower()
    selected = "bitnet-2b"
    if "gemma" in active_name:
        selected = "gemma2-2b"
    elif "llama" in active_name:
        selected = "llama3.2-3b"
        
    return JSONResponse({
        "models": models,
        "selected": selected
    })

@app.post("/api/v1/chat-model-select", tags=["System"])
async def select_chat_model(req: ChatModelSelectRequest) -> JSONResponse:
    state: AppState = get_state(app)
    try:
        await state.meta_agent.switch_model(req.model_id)
        return JSONResponse({"status": "success", "selected": req.model_id})
    except Exception as e:
        logger.error("Model switch failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _async_vlm_enrich(
    file_path: "Path",
    image_pages: list[int],
    vector_store: Any,
    vlm_id: str,
    sparse_index: Any,
) -> None:
    """
    Background task: run selected VLM on image-bearing pages.
    Dynamically instantiates the VLM, processes images sequentially,
    and unloads the model when done to free RAM.
    Uses the AppState-shared sparse_index to ensure VLM chunks are
    written to the same database that the query path reads from.
    """
    if not image_pages:
        return

    # ── Memory Governor check ────────────────────────────────────
    # CRITICAL: Ollama runs as a SEPARATE OS process and does NOT consume
    # Python in-process RAM. Use a higher ceiling (97%) for Ollama-based VLMs.
    # In-process HuggingFace VLMs (SmolVLM, Florence, QwenVL) use 85%.
    from src.modules.ragforge_indexer import MEMORY_CEILING_PCT, MEMORY_CEILING_PCT_OLLAMA
    import psutil as _psutil
    is_ollama_provider = "ollama" in vlm_id.lower()
    ceiling = MEMORY_CEILING_PCT_OLLAMA if is_ollama_provider else MEMORY_CEILING_PCT
    mem_pct = _psutil.virtual_memory().percent
    if mem_pct >= ceiling:
        logger.warning(
            "⚠️  VLM enrichment skipped for '%s' — memory at %.1f%% (ceiling: %.0f%% for %s)",
            file_path.name, mem_pct, ceiling, "Ollama" if is_ollama_provider else "in-process VLM"
        )
        return
    logger.info(
        "VLM enrichment approved for '%s' — memory %.1f%% / ceiling %.0f%% (provider: %s)",
        file_path.name, mem_pct, ceiling, vlm_id
    )
        
    try:
        from src.modules.ragforge.vlm_provider import get_vlm_provider
        vlm = get_vlm_provider(vlm_id)
        if not vlm:
            logger.error("VLM enrichment failed: unknown provider %s", vlm_id)
            return
            
        logger.info("VLM background: starting enrichment for %d pages using %s",
                    len(image_pages), vlm.name)
                    
        import fitz
        from langchain_core.documents import Document
        import uuid as uuid_mod
        
        pdf_doc = fitz.open(str(file_path))
        extracted_chunks = []
        
        for page_num in image_pages:
            if page_num >= len(pdf_doc):
                continue

            # Check memory before each page — stop early if system is under pressure
            # Uses the same Ollama-aware ceiling computed at enrichment start
            if _psutil.virtual_memory().percent >= ceiling:
                logger.warning("VLM stopping early at page %d — memory pressure (%.1f%% >= %.0f%%)",
                               page_num, _psutil.virtual_memory().percent, ceiling)
                break
                
            page = pdf_doc[page_num]
            # Use 1x resolution (144 DPI). 2x = 6MB/page wasted RAM;
            # SmolVLM/Florence downsample internally to 384/512px anyway.
            mat = fitz.Matrix(1.0, 1.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            
            # Analyze page with the VLM
            try:
                prompt = (
                    "Analyze this page. First, identify the type of content "
                    "(e.g., Academic Graph, Astrological Chart, Financial Table, Flowchart, Diagram). "
                    "Then, extract all data, labels, symbols, text, and logical relationships present "
                    "in extreme detail. Structure the output clearly. "
                    "If there are absolutely no visual figures, charts, or tables, output 'NO_FIGURES'."
                )
                analysis = await vlm.analyze_image(img_bytes, prompt)
                
                if analysis and "NO_FIGURES" not in analysis:
                    extracted_chunks.append(Document(
                        page_content=f"Visual Analysis of Page {page_num + 1}:\n{analysis}",
                        metadata={
                            "source": file_path.name,
                            "chunk_type": "vlm_analysis",
                            "page": page_num,
                            "chunk_id": str(uuid_mod.uuid4()),
                            "vlm_provider": vlm.id
                        }
                    ))
            except Exception as e:
                logger.warning("VLM failed on page %d: %s", page_num + 1, e)
                
            del pix, img_bytes
            
        pdf_doc.close()
        
        # Aggressive unload to return RAM to LLM / WatchTower
        await vlm.unload_model()
        
        if extracted_chunks:
            source_name = file_path.name
            
            # ── Tiered VLM chunk preservation ────────────────────────
            # VLM tier order: Qwen3.5 (Ultra) > Florence (Standard) > SmolVLM (Lite)
            # If a higher-tier VLM has already analyzed this source, keep its chunks
            # and discard the new lower-quality ones. This means:
            # - Switching from Qwen → SmolVLM keeps the Qwen visual analysis.
            # - Running SmolVLM then upgrading to Qwen → Qwen replaces SmolVLM.
            VLM_TIER_RANK = {
                "smolvlm-256m": 1,
                "florence-2": 2,
                "ollama-qwen3.5-9b": 3,
            }
            current_rank = VLM_TIER_RANK.get(vlm.id, 1)
            
            existing_best_tier = 0
            if sparse_index is not None:
                try:
                    # Check what VLM tier has already indexed this source
                    existing = sparse_index.get_vlm_chunks(source_name)
                    for doc in existing:
                        ex_provider = doc.metadata.get("vlm_provider", "")
                        ex_rank = VLM_TIER_RANK.get(ex_provider, 0)
                        existing_best_tier = max(existing_best_tier, ex_rank)
                except Exception:
                    pass
            
            if existing_best_tier > current_rank:
                logger.info(
                    "VLM tiering: skipping %s chunks (tier %d) — "
                    "higher-quality tier-%d analysis already exists for '%s'",
                    vlm.id, current_rank, existing_best_tier, source_name
                )
            else:
                # Delete existing VLM chunks for this source (same or lower tier)
                # and insert the new ones
                if sparse_index is not None:
                    try:
                        sparse_index.delete_vlm_chunks(source_name)
                    except Exception:
                        pass
                
                # Also clear existing VLM chunks from ChromaDB for this source
                try:
                    existing_chroma = vector_store.get(where={
                        "$and": [{"source": source_name}, {"chunk_type": "vlm_analysis"}]
                    })
                    if existing_chroma and existing_chroma.get("ids"):
                        vector_store.delete(ids=existing_chroma["ids"])
                except Exception:
                    pass
                
                # Add new VLM chunks to ChromaDB
                vector_store.add_documents(extracted_chunks)
                
                # Add to FTS5 sparse index via the shared AppState singleton
                if sparse_index is not None:
                    try:
                        sparse_index.add_documents(extracted_chunks)
                    except Exception as sparse_err:
                        logger.warning("VLM sparse index write failed: %s", sparse_err)
                
                logger.info(
                    "VLM background: enriched index with %d visual chunks (provider: %s, tier: %d)",
                    len(extracted_chunks), vlm.id, current_rank
                )
        else:
            logger.info("VLM background: no visual chunks extracted")
    except Exception as e:
        logger.error("VLM background enrichment failed: %s", e)

# ── Chat Request / Response models ────────────────────────────────
class ChatRequest(BaseModel):
    """Input model for a single chat turn."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    module: str = Field(
        default="localbuddy",
        description="Which AetherForge module to route to: ragforge | localbuddy | watchtower | streamsync | tunelab",
    )
    message: str = Field(..., min_length=1, max_length=16384)
    xray_mode: bool = Field(default=False, description="Return full causal trace")
    context: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Output model for a single chat turn."""
    session_id: str
    response: str
    module: str
    latency_ms: float
    tool_calls: list[dict[str, Any]] = []
    policy_decisions: list[dict[str, Any]] = []
    causal_graph: dict[str, Any] | None = None  # Only when xray_mode=True
    faithfulness_score: float | None = None


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint. Routes the message through the LangGraph
    meta-agent, which:
      1. Selects the appropriate module graph
      2. Runs every tool call through Silicon Colosseum
      3. Records the interaction in the replay buffer + SessionStore
      4. Returns response with optional X-Ray trace
    """
    state: AppState = get_state(app)
    t0 = time.perf_counter()
    session_store = state.session_store

    # ── Session bootstrap ─────────────────────────────────────────
    # Ensure the session exists in the DB; create it if new.
    # If the server restarted, hydrate MetaAgent memory from persisted history.
    if not session_store.session_exists(request.session_id):
        session_store.create_session(
            module=request.module,
            session_id=request.session_id,
        )
    elif request.session_id not in state.meta_agent._session_memories:
        # Server restarted — restore memory from DB so the LLM has context
        try:
            restored = session_store.to_langchain_messages(request.session_id)
            if restored:
                state.meta_agent._session_memories[request.session_id] = restored
                logger.info(
                    "Restored %d messages for session %s from SessionStore",
                    len(restored), request.session_id[:8]
                )
        except Exception as hydrate_err:
            logger.warning("Session hydration failed (non-fatal): %s", hydrate_err)

    try:
        result = await state.meta_agent.run(
            MetaAgentInput(
                session_id=request.session_id,
                module=request.module,
                message=request.message,
                xray_mode=request.xray_mode,
                context=request.context,
            )
        )
    except Exception as exc:
        logger.exception("Meta-agent error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    # ── Persist this turn to SessionStore (non-blocking) ──────────
    def _persist():
        try:
            meta = {
                "module": request.module,
                "latency_ms": round(latency_ms, 2),
                "faithfulness_score": result.faithfulness_score,
            }
            session_store.append_message(request.session_id, "user", request.message)
            session_store.append_message(request.session_id, "assistant", result.response, meta)
        except Exception as pe:
            logger.warning("Session persist failed (non-fatal): %s", pe)

    asyncio.create_task(asyncio.to_thread(_persist))

    # Record to replay buffer (async, non-blocking)
    asyncio.create_task(
        state.replay_buffer.record(
            session_id=request.session_id,
            module=request.module,
            prompt=request.message,
            response=result.response,
            tool_calls=result.tool_calls,
            faithfulness_score=result.faithfulness_score,
        )
    )

    return ChatResponse(
        session_id=request.session_id,
        response=result.response,
        module=request.module,
        latency_ms=round(latency_ms, 2),
        tool_calls=result.tool_calls,
        policy_decisions=result.policy_decisions,
        causal_graph=result.causal_graph if request.xray_mode else None,
        faithfulness_score=result.faithfulness_score,
    )


@app.get("/api/v1/modules", tags=["Modules"])
async def list_modules() -> JSONResponse:
    """Returns metadata for all available AetherForge modules."""
    return JSONResponse([
        {
            "id": "ragforge",
            "name": "RAGForge",
            "description": "Retrieval-Augmented Generation with local ChromaDB vector store",
            "icon": "search",
        },
        {
            "id": "localbuddy",
            "name": "LocalBuddy",
            "description": "Conversational AI assistant with persistent memory",
            "icon": "message-square",
        },
        {
            "id": "watchtower",
            "name": "WatchTower",
            "description": "Real-time anomaly detection and system monitoring",
            "icon": "eye",
        },
        {
            "id": "streamsync",
            "name": "StreamSync",
            "description": "Event stream processing and pattern recognition",
            "icon": "activity",
        },
        {
            "id": "tunelab",
            "name": "TuneLab",
            "description": "Interactive model fine-tuning with OPLoRA",
            "icon": "sliders",
        },
    ])


@app.get("/api/v1/policies", tags=["Guardrails"])
async def get_policies() -> JSONResponse:
    """Returns the current OPA policy source for the PolicyEditor UI."""
    state: AppState = get_state(app)
    policy_text = await state.colosseum.get_policy_source()
    return JSONResponse({"policy": policy_text})


@app.post("/api/v1/policies", tags=["Guardrails"])
async def update_policy(body: dict[str, str]) -> JSONResponse:
    """
    Hot-reload OPA policy from the PolicyEditor without restart.
    Validates Rego syntax before applying.
    """
    state: AppState = get_state(app)
    new_policy = body.get("policy", "")
    result = await state.colosseum.update_policy(new_policy)
    return JSONResponse(result)


@app.post("/api/v1/learning/trigger", tags=["Learning"])
async def trigger_training() -> JSONResponse:
    """
    Manually trigger an OPLoRA fine-tuning run.
    Used from TuneLab UI or CLI. Bypasses battery/cpu gates.
    """
    asyncio.create_task(_nightly_oplora_job(app, force=True))
    return JSONResponse({"status": "triggered", "message": "OPLoRA training job started"})


@app.get("/api/v1/replay/stats", tags=["Learning"])
async def replay_stats() -> JSONResponse:
    """Returns replay buffer statistics for the TuneLab dashboard."""
    state: AppState = get_state(app)
    stats = await state.replay_buffer.get_stats()
    return JSONResponse(stats)


@app.get("/api/v1/replay/items", tags=["Learning"])
async def replay_items(limit: int = 50) -> JSONResponse:
    """Returns the latest N records from the replay buffer for the TuneLab UI."""
    state: AppState = get_state(app)
    items = await state.replay_buffer.sample(n=limit, min_faithfulness=0.0, exclude_used=False)
    # Sort descending by timestamp so newest appear first
    items.sort(key=lambda x: x.get("timestamp_utc", 0), reverse=True)
    return JSONResponse(items)


@app.get("/api/v1/learning/capacity", tags=["Learning"])
async def learning_capacity() -> JSONResponse:
    """Returns the remaining orthogonal capacity for OPLoRA."""
    state: AppState = get_state(app)
    from src.learning.oploRA_manager import OPLoRAManager
    manager = OPLoRAManager(state.settings)
    manager.load_checkpoints()
    capacity = manager.estimate_capacity()
    return JSONResponse({
        "capacity_pct": round(capacity * 100, 1),
        "total_tasks": sum(len(v) for v in manager._subspaces.values())
    })


# ── Webhook Ingestion (WatchTower / StreamSync) ───────────────────

class WebhookPayload(BaseModel):
    """Generic payload for incoming module webhooks."""
    event: str = "webhook"
    
    # Allow arbitrary extra fields for dynamic metrics (cpu, mem, etc.)
    class Config:
        extra = "allow"

@app.post("/api/v1/events", tags=["Modules"])
async def ingest_event(background_tasks: BackgroundTasks, payload: WebhookPayload = Body(...)) -> JSONResponse:
    """
    Primary ingestion endpoint for AetherForge modules.
    - Routes raw webhooks into StreamSync's pattern matcher.
    - If numeric metrics are found (e.g. "cpu": 85), routes to WatchTower Z-Score math.
    """
    payload_dict = payload.dict(exclude_unset=True)
    
    from src.modules.streamsync.graph import emit_event
    from src.modules.watchtower.graph import ingest_metric

    source = payload_dict.pop("_source", "api")
    event_type = payload_dict.get("event", "webhook")

    # 1. Emit to StreamSync ring buffer
    event_id = emit_event(event_type=event_type, payload=payload_dict, source=source)

    # 2. Extract numeric metrics for WatchTower Z-Score anomaly detection
    for key, value in payload_dict.items():
        if isinstance(value, (int, float)):
            # Don't block the API response while calculating stats
            background_tasks.add_task(ingest_metric, key, float(value))

    return JSONResponse({"status": "ingested", "id": event_id})


@app.get("/api/v1/events/stream", tags=["Modules"])
async def stream_events(limit: int = 20) -> JSONResponse:
    """Returns the latest events from StreamSync for the Event Console UI."""
    from src.modules.streamsync.graph import _EVENT_STREAM
    # deque doesn't support slicing directly, convert to list
    events = list(_EVENT_STREAM)[-limit:]
    # Return newest first
    return JSONResponse(events[::-1])


@app.get("/api/v1/metrics/stream", tags=["Modules"])
async def stream_metrics() -> JSONResponse:
    """Returns the latest sliding-window Z-Scores from WatchTower for the Telemetry UI."""
    from src.modules.watchtower.graph import _METRIC_WINDOWS, _Z_THRESHOLD
    import numpy as np
    
    current_state = {}
    for metric, window in _METRIC_WINDOWS.items():
        if not window:
            continue
        latest_val = window[-1]
        z_score = 0.0
        is_anomaly = False
        if len(window) >= 10:
            arr = np.array(window)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std > 1e-8:
                z_score = (latest_val - mean) / std
                is_anomaly = abs(z_score) > _Z_THRESHOLD
        
        current_state[metric] = {
            "value": round(latest_val, 2),
            "z_score": round(z_score, 2),
            "is_anomaly": is_anomaly,
            "window_size": len(window)
        }
    
    return JSONResponse(current_state)


class InjectAnomalyRequest(BaseModel):
    metric: str = "mem"
    value: float = 99.5

@app.post("/api/v1/watchtower/inject_anomaly", tags=["Modules"])
async def inject_anomaly(req: InjectAnomalyRequest) -> JSONResponse:
    """
    Inject an artificial spike into WatchTower for demo/testing purposes.
    Called by the ‘⚠️ Simulate Memory Spike’ button in the frontend.
    """
    from src.modules.watchtower.graph import ingest_metric
    result = ingest_metric(req.metric, req.value)
    logger.info("Manual anomaly injected: %s=%.2f | anomaly=%s z=%.2f",
                req.metric, req.value, result["is_anomaly"], result["z_score"])
    return JSONResponse({
        "status": "injected",
        "metric": req.metric,
        "value": req.value,
        "is_anomaly": result["is_anomaly"],
        "z_score": result["z_score"],
        "baseline_locked": result.get("baseline_locked", False),
    })


class AnalyzeRequest(BaseModel):
    metric: str
    value: float
    z_score: float
    context_: dict[str, Any] = Field(default_factory=dict, alias="context")

class MitigateRequest(BaseModel):
    action: str
    target: str

@app.post("/api/v1/watchtower/analyze", tags=["Modules"])
async def trigger_rca(req: AnalyzeRequest) -> JSONResponse:
    """Trigger Root Cause Analysis for a specific anomaly."""
    state: AppState = get_state(app)
    from src.rca.root_cause_agent import RootCauseAgent
    from langchain_core.messages import SystemMessage, HumanMessage
    
    def rca_llm(prompt: str) -> str:
        # Wrap the MetaAgent's LLM to fit the RCA string->string interface
        messages = [
            SystemMessage(content="You are an expert System Administrator performing Root Cause Analysis."),
            HumanMessage(content=prompt)
        ]
        return state.meta_agent._run_llm_sync(messages, max_tokens=150)
        
    agent = RootCauseAgent(llm_fn=rca_llm, max_depth=3) # Limit depth for speed
    issue = f"{req.metric.upper()} anomaly detected: value {req.value:.1f} (Z-Score {req.z_score:.2f})"
    anomalies = [{"metric": req.metric, "value": req.value, "z_score": req.z_score}]
    
    # Run RCA synchronously in a thread so we don't block the ASGI loop
    result = await asyncio.to_thread(
        agent.analyze, issue, req.context_, [anomalies[0]]
    )
    return JSONResponse(result.to_dict())

@app.post("/api/v1/watchtower/mitigate", tags=["Modules"])
async def trigger_mitigation(req: MitigateRequest) -> JSONResponse:
    """Execute human-in-the-loop mitigation action (simulated syscall)."""
    logger.info("Executing WatchTower mitigation: action='%s' target='%s'", req.action, req.target)
    
    # Simulate syscall execution delay
    await asyncio.sleep(0.5)
    
    # Reset WatchTower baseline metrics to simulate the anomaly being immediately resolved
    # by the sysadmin's mitigation action.
    from src.modules.watchtower.graph import _METRIC_WINDOWS
    if req.action in ["Kill Process", "Restart Module"]:
        for k in _METRIC_WINDOWS:
             # Fast drop to prevent lingering anomalies in the frontend sparklines
            _METRIC_WINDOWS[k].clear()
            for _ in range(10): _METRIC_WINDOWS[k].append(45.0 if k != 'net' else 12.0)
            
    return JSONResponse({"status": "success", "message": f"Action '{req.action}' executed on {req.target}"})


# ── WebSocket: Streaming Chat ─────────────────────────────────────
@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket endpoint for streaming token-by-token responses.
    The frontend opens this once per session and sends/receives JSON.

    Message format (client → server):
      {"message": "...", "module": "localbuddy", "xray_mode": false}

    Message format (server → client, streamed):
      {"type": "token", "content": "Hello"} × N
      {"type": "done", "latency_ms": 123.4, "policy_decisions": [...]}
    """
    state: AppState = get_state(app)
    await websocket.accept()
    logger.info("WebSocket open: session=%s", session_id)

    try:
        while True:
            data = await websocket.receive_json()
            module = data.get("module", "localbuddy")
            message = data.get("message", "")
            xray_mode = data.get("xray_mode", False)

            if not message:
                await websocket.send_json({"type": "error", "content": "Empty message"})
                continue

            t0 = time.perf_counter()

            # Stream tokens via async generator from meta-agent
            async for chunk in state.meta_agent.stream(
                MetaAgentInput(
                    session_id=session_id,
                    module=module,
                    message=message,
                    xray_mode=xray_mode,
                    context={},
                )
            ):
                await websocket.send_json(chunk)

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            await websocket.send_json({"type": "done", "latency_ms": latency_ms})

    except WebSocketDisconnect:
        logger.info("WebSocket closed: session=%s", session_id)


# ── Background Jobs ───────────────────────────────────────────────
async def _nightly_oplora_job(app_: FastAPI, force: bool = False) -> None:
    """
    Nightly OPLoRA training job. Only runs if:
      - Battery > min_battery_pct (default 30%)
      - System is not under heavy CPU load (< 50%)
    Bypassed if force=True.
    """
    state: AppState = get_state(app_)
    settings = state.settings

    if not force:
        # ── Battery gate ──────────────────────────────────────────────
        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged:
            if battery.percent < settings.oploра_min_battery_pct:
                logger.info(
                    "Nightly OPLoRA skipped: battery %.0f%% < threshold %.0f%%",
                    battery.percent,
                    settings.oploра_min_battery_pct,
                )
                return

        # ── CPU gate ──────────────────────────────────────────────────
        cpu_load = psutil.cpu_percent(interval=1.0)
        if cpu_load > 60.0:
            logger.info("Nightly OPLoRA skipped: CPU at %.0f%%", cpu_load)
            return

    logger.info("Starting %sOPLoRA fine-tuning...", "forced " if force else "nightly ")
    try:
        from src.learning.bitnet_trainer import BitNetTrainer
        trainer = BitNetTrainer(settings, state.replay_buffer)
        await trainer.run_oploora_cycle()
        logger.info("%sOPLoRA complete.", "Forced " if force else "Nightly ")
    except Exception as exc:
        logger.exception("OPLoRA failed: %s", exc)


# ════════════════════════════════════════════════════════════════
# MULTI NODE SYNC API
# ════════════════════════════════════════════════════════════════

@app.websocket("/api/v1/sync/ws/{peer_node_id}")
async def sync_websocket_endpoint(websocket: WebSocket, peer_node_id: str):
    """
    WebSocket endpoint for incoming P2P CRDT sync connections.
    """
    state: AppState = websocket.app.state.app_state
    if not state.sync_manager:
        await websocket.close(code=1011, reason="Sync engine offline")
        return
    await state.sync_manager.handle_websocket(websocket, peer_node_id)

@app.get("/api/v1/sync/info", tags=["Sync"])
async def sync_get_info(request: Request) -> JSONResponse:
    """Returns the Node ID, active peers, and QR connection string."""
    state: AppState = request.app.state.app_state
    if not state.sync_manager:
        return JSONResponse({"status": "offline"})

    # In a real app the ephemeral key would be bound to a short-lived memory session.
    # For now, we generate one to encode in the QR.
    from src.modules.sync.crypto import SyncCrypto
    ephemeral_key = SyncCrypto.generate_key()
    
    # Authorize it locally for when the peer scans it
    uri = SyncCrypto.create_pairing_uri(
        ip=state.sync_manager.discovery.get_local_ip(),
        port=state.sync_manager.port,
        node_id=state.sync_manager.node_id,
        key_b64=ephemeral_key
    )

    return JSONResponse({
        "status": "online",
        "node_id": state.sync_manager.node_id,
        "pairing_uri": uri,
        "ephemeral_key": ephemeral_key,
        "peers": state.sync_manager.discovery.get_active_peers(),
        "authorized": list(state.sync_manager.authorized_peers.keys())
    })

@app.post("/api/v1/sync/pair", tags=["Sync"])
async def sync_pair_device(request: Request) -> JSONResponse:
    """Pairs a new device using a scanned pairing URI."""
    state: AppState = request.app.state.app_state
    if not state.sync_manager:
        return JSONResponse({"status": "offline"}, status_code=500)

    try:
        body = await request.json()
        uri = body.get("uri")
        if not uri:
            return JSONResponse({"error": "Missing 'uri' payload"}, status_code=400)

        from src.modules.sync.crypto import SyncCrypto
        parsed = SyncCrypto.parse_pairing_uri(uri)
        
        peer_node_id = parsed["node_id"]
        key_b64 = parsed["key"]
        
        # Authorize the peer so we can accept its websockets
        state.sync_manager.authorize_peer(peer_node_id, key_b64)

        return JSONResponse({
            "status": "paired",
            "peer": peer_node_id
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ── CLI Entry (typer) ─────────────────────────────────────────────
cli_app = typer.Typer(name="aetherforge", add_completion=False)


@cli_app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8765, help="Bind port"),
    reload: bool = typer.Option(False, help="Hot-reload (dev only)"),
    workers: int = typer.Option(1, help="Number of workers"),
) -> None:
    """Start the AetherForge FastAPI server."""
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=get_settings().aetherforge_log_level,
    )


if __name__ == "__main__":
    cli_app()
