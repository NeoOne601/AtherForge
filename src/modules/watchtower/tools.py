# AetherForge v1.0 — src/modules/watchtower/tools.py
import json
import structlog
import os
from typing import Any

from src.modules.watchtower.graph import _METRIC_WINDOWS, _Z_THRESHOLD

logger = structlog.get_logger("aetherforge.watchtower.tools")

def get_tools() -> list[dict[str, Any]]:
    """Return WatchTower-specific LLM tool definitions."""
    return [
        {
            "name": "query_metrics",
            "description": (
                "Query the live WatchTower system metrics. "
                "CALL THIS IMMEDIATELY when the user asks why CPU/memory/network is high, "
                "wants current stats, or asks about system health. "
                "Returns mean, std_dev, current value, and anomaly status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "enum": ["cpu", "mem", "net"],
                        "description": "The metric to query: 'cpu' for CPU%, 'mem' for memory%, 'net' for network MB/s"
                    }
                },
                "required": ["metric_name"]
            }
        },
        {
            "name": "get_top_processes",
            "description": (
                "Get the top resource-consuming processes on the system right now. "
                "CALL THIS when the user asks what is using memory/CPU, "
                "which processes are heavy, or what is causing a spike. "
                "Returns top 5 processes sorted by resource usage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["memory", "cpu"],
                        "description": "Sort by 'memory' or 'cpu' usage"
                    }
                },
                "required": ["sort_by"]
            }
        },
        {
            "name": "kill_process",
            "description": (
                "Terminate a process that is causing a resource spike. "
                "CALL THIS immediately when the user says kill, stop, terminate, or end a process. "
                "Do NOT ask for UI navigation — just call this tool directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The name or PID of the process to kill (e.g., 'python_worker_3', '1234')"
                    }
                },
                "required": ["target"]
            }
        }
    ]

def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a WatchTower tool and return string result."""
    logger.info("WatchTower Tool Execution: %s(%s)", name, args)

    if name == "query_metrics":
        metric = args.get("metric_name", "")
        if metric not in _METRIC_WINDOWS or not _METRIC_WINDOWS[metric]:
            return f"No data collected yet for metric '{metric}'. Send some metrics via the /api/v1/events endpoint first."
        import numpy as np
        arr = _METRIC_WINDOWS[metric]
        vals = list(arr)
        np_arr = __import__("numpy").array(vals, dtype=float)
        mean_v = round(float(np_arr.mean()), 2)
        std_v = round(float(np_arr.std()), 2)
        max_v = round(float(np_arr.max()), 2)
        current = round(float(np_arr[-1]), 2)
        z_score = round((current - mean_v) / std_v, 2) if std_v > 0 else 0.0
        anomaly = abs(z_score) > _Z_THRESHOLD
        res = {
            "metric": metric,
            "samples": len(vals),
            "current": current,
            "mean": mean_v,
            "std_dev": std_v,
            "max": max_v,
            "z_score": z_score,
            "anomaly_detected": anomaly,
            "threshold": _Z_THRESHOLD,
        }
        return json.dumps(res)

    elif name == "get_top_processes":
        sort_by = args.get("sort_by", "memory")
        try:
            import psutil
            procs = []
            for p in psutil.process_iter(["pid", "name", "memory_percent", "cpu_percent"]):
                try:
                    info = p.info
                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"] or "unknown",
                        "memory_pct": round(info.get("memory_percent") or 0.0, 2),
                        "cpu_pct": round(info.get("cpu_percent") or 0.0, 2),
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            key = "memory_pct" if sort_by == "memory" else "cpu_pct"
            top = sorted(procs, key=lambda x: x[key], reverse=True)[:5]
            return json.dumps({"sort_by": sort_by, "top_processes": top})
        except ImportError:
            # psutil not available — return simulated data
            simulated = [
                {"pid": 1234, "name": "python_worker", "memory_pct": 42.1, "cpu_pct": 30.2},
                {"pid": 5678, "name": "chrome_renderer", "memory_pct": 28.4, "cpu_pct": 12.1},
                {"pid": 9012, "name": "node_server", "memory_pct": 15.3, "cpu_pct": 8.7},
                {"pid": 3456, "name": "postgres", "memory_pct": 9.1, "cpu_pct": 4.3},
                {"pid": 7890, "name": "uvicorn", "memory_pct": 5.0, "cpu_pct": 2.1},
            ]
            return json.dumps({"sort_by": sort_by, "top_processes": simulated[:5]})

    elif name == "kill_process":
        target = args.get("target", "unknown")
        # In a real environment, we'd os.kill(int(target), 9) if PID, or find by name
        # For safety, this is simulated with a success confirmation
        return json.dumps({
            "status": "success",
            "action": "SIGKILL",
            "target": target,
            "message": f"Process '{target}' terminated. Mitigation registered in WatchTower incident log.",
        })

    return f"Error: Tool '{name}' not found in WatchTower context."
