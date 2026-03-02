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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

    # ── 3. Initialise Silicon Colosseum (OPA + FSM) ───────────────
    state.colosseum = SiliconColosseum(settings)
    await state.colosseum.initialize()
    logger.info("Silicon Colosseum guardrails active | mode=%s", settings.opa_mode)

    # ── 4. Initialise LangGraph Meta-Agent (loads BitNet model) ───
    state.meta_agent = MetaAgent(settings, state.colosseum)
    await state.meta_agent.initialize()
    logger.info("Meta-agent ready | modules=5 | model=%s", settings.bitnet_model_path.name)

    # ── 5. Start nightly OPLoRA scheduler ─────────────────────────
    state.scheduler = AsyncIOScheduler(timezone="local")
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

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("AetherForge shutting down...")
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
    Used from TuneLab UI or CLI. Checks battery before starting.
    """
    asyncio.create_task(_nightly_oplora_job(app))
    return JSONResponse({"status": "triggered", "message": "OPLoRA training job started"})


@app.get("/api/v1/replay/stats", tags=["Learning"])
async def replay_stats() -> JSONResponse:
    """Returns replay buffer statistics for the TuneLab dashboard."""
    state: AppState = get_state(app)
    stats = await state.replay_buffer.get_stats()
    return JSONResponse(stats)


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
async def _nightly_oplora_job(app_: FastAPI) -> None:
    """
    Nightly OPLoRA training job. Only runs if:
      - Battery > min_battery_pct (default 30%)
      - System is not under heavy CPU load (< 50%)

    This is called by APScheduler at the configured time, and can
    also be triggered via POST /api/v1/learning/trigger.
    """
    state: AppState = get_state(app_)
    settings = state.settings

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
    if cpu_load > 50.0:
        logger.info("Nightly OPLoRA skipped: CPU at %.0f%%", cpu_load)
        return

    logger.info("Starting nightly OPLoRA fine-tuning...")
    try:
        from src.learning.bitnet_trainer import BitNetTrainer
        trainer = BitNetTrainer(settings, state.replay_buffer)
        await trainer.run_oploora_cycle()
        logger.info("Nightly OPLoRA complete.")
    except Exception as exc:
        logger.exception("Nightly OPLoRA failed: %s", exc)


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
