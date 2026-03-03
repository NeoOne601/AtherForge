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
import logging
import time
import uuid
from contextlib import asynccontextmanager
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
from src.learning.replay_buffer import ReplayBuffer
from src.meta_agent import MetaAgent, MetaAgentInput

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
    startup_ms: float


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
    # BAAI/bge-m3 replaces all-MiniLM-L6-v2:
    #   - 8192 token limit (vs 512) — full PDF sections fit in one vector
    #   - 1024 dimensions (vs 384) — richer semantic space for academic text
    #   - Dense + Sparse + Multi-vector unified model (best retrieval accuracy)
    #   - Fully offline, ~570MB, Apache 2.0
    logger.info("Initializing ChromaDB with BAAI/bge-m3 embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},          # works on any machine, MPS auto-detected
        encode_kwargs={"normalize_embeddings": True},  # required for cosine similarity
    )
    state.vector_store = Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
    )
    logger.info("ChromaDB active at %s | model=bge-m3 | dims=1024 | max_tokens=8192",
                settings.chroma_path)

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

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────
    logger.info("AetherForge shutting down...")
    _poller_task.cancel()
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

@app.get("/health", tags=["System"])
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


@app.post("/api/v1/ragforge/upload", tags=["RAGForge"])
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """
    Ingest a document (PDF, MD, CSV, TXT) into the local vector DB.
    Triggered when a user drags and drops a file into the RAGForge Vault.
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

        # Vectorize and index in a background thread to avoid blocking the event loop
        from src.modules.ragforge_indexer import index_document
        result = await asyncio.to_thread(index_document, file_path, state.vector_store)

        # result is a dict: {file, chunks_added, parser, chunk_breakdown}
        chunks_added = result.get("chunks_added", 0) if isinstance(result, dict) else int(result)
        parser = result.get("parser", "unknown") if isinstance(result, dict) else "legacy"
        breakdown = result.get("chunk_breakdown", {}) if isinstance(result, dict) else {}

        logger.info(
            "Indexed '%s' — %d chunks via %s | breakdown=%s",
            file.filename, chunks_added, parser, breakdown,
        )
        return JSONResponse({
            "status": "success",
            "chunks_indexed": chunks_added,
            "parser": parser,
            "chunk_breakdown": breakdown,
            "file": file.filename,
        })
    except Exception as e:
        logger.exception("Failed to process RAG document %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e)) from e


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
      3. Records the interaction in the replay buffer
      4. Returns response with optional X-Ray trace
    """
    state: AppState = get_state(app)
    t0 = time.perf_counter()

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
    result = await asyncio.get_event_loop().run_in_executor(
        None, 
        lambda: agent.analyze(issue=issue, context=req.context_, anomalies=anomalies)
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
