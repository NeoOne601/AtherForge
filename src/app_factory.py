import asyncio
import os
import time
import uuid

import structlog
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

# ── 0. Load Environment BEFORE ML Imports ───────────────────────
# This ensures that HF_HOME and other cache variables from .env
# are injected into os.environ before huggingface_hub initializes.
from dotenv import load_dotenv

load_dotenv()

# ── 0b. Apply user-saved settings.json overrides ────────────────
# These take priority over .env defaults for paths like HF_HOME.
_settings_file = os.path.join("data", "settings.json")
if os.path.exists(_settings_file):
    try:
        import json as _json
        with open(_settings_file) as _f:
            for _k, _v in _json.load(_f).items():
                os.environ[_k] = str(_v)
    except Exception:
        pass  # Non-fatal: fall back to .env defaults

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings
from src.logging_setup import setup_logging
from src.routers import (
    chat_router,
    export_router,
    guardrails_router,
    learning_router,
    ragforge_router,
    ragforge_stream_router,
    ragforge_tree_router,
    sessions_router,
    settings_router,
    streamsync_router,
    sync_router,
    watchtower_router,
)
from src.schemas import AppState
from src.utils import safe_create_task

logger = structlog.get_logger("aetherforge.factory")


async def _nightly_oplora_job(app: FastAPI, force: bool = False) -> None:
    """Nightly background job for OPLoRA fine-tuning."""
    state: AppState = app.state.app_state
    logger.info("OPLoRA job starting", force=force)
    try:
        from src.learning.oplora_manager import OPLoRAManager

        manager = OPLoRAManager(state.settings)
        await manager.run_evolution_loop(state.replay_buffer, state.history_manager)
        logger.info("OPLoRA job complete")
    except Exception as e:
        logger.error("OPLoRA job failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    t0 = time.perf_counter()
    from src.core.container import container

    settings = get_settings()

    # ── 1. Initialize Structured Logging ───────────────────────
    queue_handler = setup_logging(env=settings.aetherforge_env)
    queue_handler.loop = asyncio.get_running_loop()
    logger.info(
        "AetherForge starting", env=settings.aetherforge_env, port=settings.aetherforge_port
    )

    state = AppState()
    app.state.app_state = state

    # ── 2. Initialize Core Services via Container ────────────────
    await container.initialize_all(state)

    # ── 3. Background Utility Tasks ─────────────────────────────
    # Location (Background)
    async def _fetch_location() -> None:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get("https://ipinfo.io/json")
                if r.status_code == 200:
                    d = r.json()
                    state.system_location = (
                        f"{d.get('city')}, {d.get('region')}, {d.get('country')}"
                    )
                    logger.info("System Location Verified", location=state.system_location)
        except Exception as e:
            logger.warning("Location retrieval failed", error=str(e))

    safe_create_task(_fetch_location(), name="fetch_location")

    # ── 4. Sync Management ───────────────────────────────────────────
    # (Managed by src.core.container)

    # ── 5. Module Background Tasks ──────────────────────────────
    from src.modules.streamsync.directory_watcher import (
        StreamSyncDirectoryWatcher as DirectoryWatcher,
    )
    from src.modules.streamsync.rss_feeder import rss_poller_task

    # RSS
    rss_state_file = settings.data_dir / "streamsync_rss_feeds.json"
    if rss_state_file.exists():
        try:
            with open(rss_state_file) as f:
                import json

                state.streamsync_rss_feeds = json.load(f).get("feeds", [])
        except Exception:
            pass

    state.rss_task = safe_create_task(rss_poller_task(app), name="rss_poller")

    # Directory Watcher
    live_folder = settings.data_dir.resolve() / "LiveFolder"
    state.directory_watcher = DirectoryWatcher(
        live_folder,
        asyncio.get_running_loop(),
        state,
    )
    state.directory_watcher.start()

    # Scheduler
    state.scheduler = AsyncIOScheduler()
    state.scheduler.add_job(
        _nightly_oplora_job, CronTrigger(hour=3), args=[app], id="nightly_oplora"
    )
    # Retry skipped VLM enrichment every 5 minutes
    state.scheduler.add_job(
        container.get_service("document_intelligence").retry_pending_ocr,
        "interval",
        minutes=5,
        id="vlm_retry_job",
    )
    state.scheduler.start()

    state.startup_ms = (time.perf_counter() - t0) * 1000
    msg_ms = f"{state.startup_ms:.2f}"
    logger.info("AetherForge startup complete", duration_ms=float(msg_ms))

    yield

    # ── Shutdown ──────────────────────────────────────────
    logger.info("AetherForge shutting down")
    state.rss_task.cancel()
    if state.directory_watcher:
        state.directory_watcher.stop()
    if state.sync_manager:
        await state.sync_manager.stop()
    state.scheduler.shutdown(wait=False)
    await container.shutdown_all()


def create_app() -> FastAPI:
    app = FastAPI(title="AetherForge", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:1420", "http://127.0.0.1:1420", "https://tauri.localhost"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    )

    # Register Routers
    app.include_router(chat_router)
    app.include_router(ragforge_router)
    app.include_router(ragforge_stream_router)
    app.include_router(ragforge_tree_router)
    app.include_router(streamsync_router)
    app.include_router(watchtower_router)
    app.include_router(sessions_router)
    app.include_router(export_router)
    app.include_router(guardrails_router)
    app.include_router(learning_router)
    app.include_router(sync_router)
    app.include_router(settings_router)

    # Mount generated files directory for charts and reports
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    generated_dir = Path(os.environ.get("DATA_DIR", "data")) / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/api/v1/generated", StaticFiles(directory=str(generated_dir)), name="generated")

    # ── Observability Middleware ───────────────────────────
    class ObservabilityMiddleware(BaseHTTPMiddleware):
        async def dispatch(
            self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            request_id = str(uuid.uuid4())
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(request_id=request_id)

            start_time = time.perf_counter()
            response = await call_next(request)
            process_time = time.perf_counter() - start_time

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)

            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                startup_ms=float(f"{app.state.app_state.startup_ms:.2f}"),
                duration=float(f"{process_time:.4f}"),
                status_code=response.status_code,
            )
            return response

    app.add_middleware(ObservabilityMiddleware)

    return app
