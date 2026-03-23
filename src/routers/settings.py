# AetherForge v1.0 — src/routers/settings.py
# ─────────────────────────────────────────────────────────────────
# Settings router: manages user-configurable dependency paths
# and application preferences. Persists to data/settings.json.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["settings"])

SETTINGS_FILE = Path("data/settings.json")

# ── Settings Schema ───────────────────────────────────────────────

SETTINGS_SCHEMA: dict[str, dict[str, Any]] = {
    "ai_models": {
        "label": "🤖 AI Models",
        "description": "Paths to LLM model files and Vision Language Models",
        "fields": {
            "BITNET_MODEL_PATH": {
                "label": "BitNet Model Path",
                "description": "Path to the BitNet GGUF model file",
                "type": "path",
                "default": "/Volumes/Apple/AI Model/bitnet-model.gguf",
            },
            "APPLE_VLM_MODEL_PATH": {
                "label": "Vision LM (VLM) Path",
                "description": "Directory for Apple Silicon VLM models",
                "type": "path",
                "default": "/Volumes/Apple/AI Model",
            },
        },
    },
    "cache_downloads": {
        "label": "📦 Cache & Downloads",
        "description": "Redirect AI model downloads to an external drive to save internal disk space",
        "fields": {
            "HF_HOME": {
                "label": "HuggingFace Home",
                "description": "Root cache directory for all HuggingFace downloads",
                "type": "path",
                "default": "/Volumes/Apple/AI Model/hf_cache",
            },
            "SENTENCE_TRANSFORMERS_HOME": {
                "label": "Sentence Transformers Cache",
                "description": "Cache for sentence-transformers embedding models (~90MB each)",
                "type": "path",
                "default": "/Volumes/Apple/AI Model/hf_cache/sentence_transformers",
            },
            "TORCH_HOME": {
                "label": "PyTorch Cache",
                "description": "Cache for PyTorch model downloads and checkpoints",
                "type": "path",
                "default": "/Volumes/Apple/AI Model/hf_cache/torch",
            },
            "DOCLING_CACHE_DIR": {
                "label": "Docling Cache",
                "description": "Cache for Docling document parsing models (~500MB)",
                "type": "path",
                "default": "/Volumes/Apple/AI Model/hf_cache/docling",
            },
        },
    },
    "data_storage": {
        "label": "💾 Data Storage",
        "description": "Where AetherForge stores its databases, indexes, and session history",
        "fields": {
            "DATA_DIR": {
                "label": "Data Directory",
                "description": "Root directory for all AetherForge application data",
                "type": "path",
                "default": "./data",
            },
            "CHROMA_PATH": {
                "label": "ChromaDB Path",
                "description": "Vector database storage for RAGForge embeddings",
                "type": "path",
                "default": "./data/chroma",
            },
            "REPLAY_BUFFER_PATH": {
                "label": "Replay Buffer Path",
                "description": "Parquet dataset for OPLoRA self-optimization training data",
                "type": "path",
                "default": "./data/replay_buffer.parquet",
            },
        },
    },
    "server": {
        "label": "⚙️ Server",
        "description": "Backend server configuration and LLM inference parameters",
        "fields": {
            "AETHERFORGE_PORT": {
                "label": "Server Port",
                "description": "Port number for the backend API server",
                "type": "number",
                "default": 8765,
                "min": 1024,
                "max": 65535,
            },
            "AETHERFORGE_LOG_LEVEL": {
                "label": "Log Level",
                "description": "Verbosity of backend logging output",
                "type": "select",
                "options": ["debug", "info", "warning", "error"],
                "default": "info",
            },
            "BITNET_N_THREADS": {
                "label": "CPU Threads",
                "description": "Number of CPU threads for LLM inference",
                "type": "number",
                "default": 8,
                "min": 1,
                "max": 64,
            },
            "BITNET_TEMPERATURE": {
                "label": "Temperature",
                "description": "LLM creativity/randomness (0.0 = deterministic, 2.0 = very creative)",
                "type": "number",
                "default": 0.7,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
            },
            "BITNET_MAX_TOKENS": {
                "label": "Max Tokens",
                "description": "Maximum tokens the LLM can generate per response",
                "type": "number",
                "default": 1024,
                "min": 64,
                "max": 8192,
            },
            "VISUAL_THEME": {
                "label": "Visual Theme",
                "description": "Choose the look and feel of the AetherForge interface",
                "type": "select",
                "options": [
                    "Sovereign Dark",
                    "Nordic Frost",
                    "Neon Cyberpunk",
                    "Monochrome Pro",
                    "Forest Terminal",
                ],
                "default": "Sovereign Dark",
            },
        },
    },
    "learning": {
        "label": "🧠 Neural Learning",
        "description": "Configuration for perpetual learning (OPLoRA and SONA)",
        "fields": {
            "OPLORA_NIGHTLY_HOUR": {
                "label": "Nightly Training Hour",
                "description": "Hour of the day to run batch OPLoRA training (0-23)",
                "type": "number",
                "default": 3,
                "min": 0,
                "max": 23,
            },
            "OPLORA_NIGHTLY_MINUTE": {
                "label": "Nightly Training Minute",
                "description": "Minute of the hour to run batch OPLoRA training (0-59)",
                "type": "number",
                "default": 0,
                "min": 0,
                "max": 59,
            },
        },
    },
}


def _load_saved_settings() -> dict[str, Any]:
    """Load user-saved settings from data/settings.json, or return empty dict."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _get_current_values() -> dict[str, dict[str, Any]]:
    """Build the full settings response with current values."""
    saved = _load_saved_settings()
    result: dict[str, Any] = {}

    for group_key, group_def in SETTINGS_SCHEMA.items():
        fields_out: dict[str, Any] = {}
        for field_key, field_def in group_def["fields"].items():
            # Priority: saved settings > env var > schema default
            current_value = saved.get(field_key)
            if current_value is None:
                current_value = os.environ.get(field_key)
            if current_value is None:
                current_value = field_def["default"]

            fields_out[field_key] = {
                **field_def,
                "value": current_value,
                "is_saved": field_key in saved,
            }

        result[group_key] = {
            "label": group_def["label"],
            "description": group_def["description"],
            "fields": fields_out,
        }

    return result


# ── Endpoints ─────────────────────────────────────────────────────


@router.get("/settings")
async def get_settings():
    """Return all settings grouped by category with current values."""
    return _get_current_values()


class SettingsUpdate(BaseModel):
    settings: dict[str, Any]


@router.put("/settings")
async def save_settings(body: SettingsUpdate):
    """Persist user settings to data/settings.json."""
    # Validate that all keys are recognized
    valid_keys = set()
    for group_def in SETTINGS_SCHEMA.values():
        for field_key in group_def["fields"]:
            valid_keys.add(field_key)

    unknown_keys = set(body.settings.keys()) - valid_keys
    if unknown_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown settings keys: {', '.join(unknown_keys)}",
        )

    # Merge with existing saved settings (don't lose keys not in this update)
    existing = _load_saved_settings()
    existing.update(body.settings)

    # Persist
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    return {
        "status": "saved",
        "restart_required": True,
        "message": "Settings saved. Restart AetherForge for changes to take full effect.",
    }


class PathValidation(BaseModel):
    path: str


@router.post("/settings/validate-path")
async def validate_path(body: PathValidation):
    """Check if a path exists and is writable."""
    p = Path(os.path.expandvars(os.path.expanduser(body.path)))

    result: dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "is_directory": p.is_dir() if p.exists() else None,
        "is_file": p.is_file() if p.exists() else None,
        "writable": False,
        "parent_exists": p.parent.exists(),
    }

    # Check writability
    if p.exists():
        result["writable"] = os.access(str(p), os.W_OK)
    elif p.parent.exists():
        result["writable"] = os.access(str(p.parent), os.W_OK)

    return result


@router.get("/settings/disk-usage")
async def get_disk_usage():
    """Return disk usage statistics for each configured path."""
    saved = _load_saved_settings()
    usage: dict[str, Any] = {}

    # Collect all path-type settings
    for group_def in SETTINGS_SCHEMA.values():
        for field_key, field_def in group_def["fields"].items():
            if field_def["type"] != "path":
                continue

            raw = saved.get(field_key) or os.environ.get(field_key) or field_def["default"]
            p = Path(os.path.expandvars(os.path.expanduser(str(raw))))

            # Find the mount point's disk usage
            target = p if p.exists() else p.parent
            try:
                disk = shutil.disk_usage(str(target))
                # Also calculate size of the specific directory
                dir_size = 0
                if p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file():
                            try:
                                dir_size += f.stat().st_size
                            except OSError:
                                pass

                usage[field_key] = {
                    "path": str(p),
                    "exists": p.exists(),
                    "dir_size_gb": round(dir_size / (1024**3), 2),
                    "disk_total_gb": round(disk.total / (1024**3), 1),
                    "disk_used_gb": round(disk.used / (1024**3), 1),
                    "disk_free_gb": round(disk.free / (1024**3), 1),
                    "disk_pct": round(disk.used / disk.total * 100, 1) if disk.total else 0,
                }
            except OSError:
                usage[field_key] = {
                    "path": str(p),
                    "exists": False,
                    "error": "Cannot access path",
                }

    return usage
