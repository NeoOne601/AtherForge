# AetherForge v1.0 — src/main.py
from __future__ import annotations
import os as _os
import typer
import uvicorn
from fastapi import FastAPI
from typing import Any

# Redirects MUST be set before any HF/torch/docling imports to ensure correct volume mapping
_EXTERNAL_AI_DRIVE = _os.environ.get("HF_HOME", "/Volumes/Apple/AI Model/hf_cache")
_hf_hub = f"{_EXTERNAL_AI_DRIVE}/hub"
_os.environ.setdefault("HF_HOME", _EXTERNAL_AI_DRIVE)
_os.environ.setdefault("HF_HUB_CACHE", _hf_hub)
_os.environ.setdefault("TRANSFORMERS_CACHE", _hf_hub)
_os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", f"{_EXTERNAL_AI_DRIVE}/sentence_transformers")
_os.environ.setdefault("TORCH_HOME", f"{_EXTERNAL_AI_DRIVE}/torch")
_os.environ.setdefault("DOCLING_CACHE_DIR", f"{_EXTERNAL_AI_DRIVE}/docling")

from src.app_factory import create_app

app = create_app()

def get_state(app: FastAPI) -> Any:
    """Return the shared AppState stored in the FastAPI application.

    Modules like rss_feeder import this helper to access the state without
    creating a circular import.
    """
    return getattr(app.state, "app_state", None)

cli_app = typer.Typer(no_args_is_help=True)

@cli_app.command()
def serve(port: int = 8765, host: str = "127.0.0.1", reload: bool = False):
    """Run the AetherForge FastAPI backend."""
    uvicorn.run("src.main:app", host=host, port=port, reload=reload)

@cli_app.command()
def version():
    """Print the AetherForge version."""
    print("AetherForge v1.0.0")

if __name__ == "__main__":
    cli_app()
