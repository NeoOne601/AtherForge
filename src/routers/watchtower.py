import time

import psutil
import structlog
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.logging_setup import LOG_QUEUE

router = APIRouter(prefix="/api/v1", tags=["System"])
logger = structlog.get_logger("aetherforge.watchtower")


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    state = request.app.state.app_state

    # Check LLM
    llm_status = "uninitialized"
    if hasattr(state, "meta_agent") and state.meta_agent._llm:
        llm_status = "ready"

    # Check Vector DB
    vector_status = "error"
    try:
        if hasattr(state, "vector_store") and state.vector_store:
            # Note: Internal _client use as in main.py
            state.vector_store._client.heartbeat()
            vector_status = "ready"
    except Exception:
        vector_status = "unreachable"

    # Check Guardrails
    guardrail_status = "ready"
    if hasattr(state, "colosseum") and state.settings.opa_mode == "server":
        try:
            import httpx

            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"{state.settings.opa_server_url}/health")
                guardrail_status = "ready" if resp.status_code == 200 else "unhealthy"
        except Exception:
            guardrail_status = "unreachable"

    overall_status = "ok"
    if "unreachable" in [vector_status, guardrail_status] or llm_status == "uninitialized":
        overall_status = "degraded"

    return JSONResponse(
        {
            "status": overall_status,
            "timestamp": time.time(),
            "dependencies": {
                "llm": llm_status,
                "vector_db": vector_status,
                "guardrails": guardrail_status,
            },
        }
    )


@router.websocket("/system/logs")
async def stream_system_logs(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Logger client connected")
    try:
        while True:
            log_entry = await LOG_QUEUE.get()
            await websocket.send_json(log_entry)
    except WebSocketDisconnect:
        logger.info("Logger client disconnected")
    except Exception as e:
        logger.error("Error streaming logs: %s", e)
        try:
            await websocket.close()
        except:
            pass


@router.get("/status")
async def system_status() -> JSONResponse:
    battery = psutil.sensors_battery()
    return JSONResponse(
        {
            "battery_pct": battery.percent if battery else None,
            "battery_plugged": battery.power_plugged if battery else None,
            "cpu_pct": psutil.cpu_percent(interval=0.1),
            "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
            "modules": ["ragforge", "localbuddy", "watchtower", "streamsync", "tunelab"],
        }
    )


@router.get("/modules")
async def list_modules() -> JSONResponse:
    return JSONResponse(
        {
            "modules": [
                {
                    "id": "localbuddy",
                    "name": "LocalBuddy",
                    "icon": "bot",
                    "description": "LLM Chat",
                },
                {
                    "id": "ragforge",
                    "name": "RAGForge",
                    "icon": "database",
                    "description": "Document Q&A",
                },
                {
                    "id": "watchtower",
                    "name": "WatchTower",
                    "icon": "activity",
                    "description": "Performance Metrics",
                },
                {
                    "id": "streamsync",
                    "name": "StreamSync",
                    "icon": "refresh",
                    "description": "Live Folders",
                },
                {
                    "id": "tunelab",
                    "name": "TuneLab",
                    "icon": "tool",
                    "description": "Offline Learning",
                },
            ]
        }
    )


@router.get("/metrics/stream")
async def get_metrics_stream() -> JSONResponse:
    """Returns real-time system metrics for HUD display."""
    import psutil

    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        net_io = psutil.net_io_counters()
        # Mocking network throughput since we don't have a history for diffing in this simple GET
        net_val = round(
            (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024 * 10), 2
        )  # MB/10s approximation

        return JSONResponse(
            {
                "cpu": {"value": cpu, "z_score": 0.0, "is_anomaly": cpu > 90},
                "mem": {"value": mem, "z_score": 0.0, "is_anomaly": mem > 90},
                "net": {"value": net_val, "z_score": 0.0, "is_anomaly": False},
            }
        )
    except Exception as e:
        logger.error("Error in metrics stream: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
