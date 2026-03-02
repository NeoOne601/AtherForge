# AetherForge v1.0 — src/config.py
# ─────────────────────────────────────────────────────────────────
# Pydantic v2 settings model. Single source of truth for all runtime
# configuration. Values cascade: .env file → env vars → defaults.
# Every setting is typed, validated, and documented inline.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AetherForgeSettings(BaseSettings):
    """
    Central configuration object for AetherForge v1.0.

    Loaded once at startup via get_settings() which is @lru_cache'd
    so the cost of .env parsing is paid only once.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core Server ───────────────────────────────────────────────
    aetherforge_env: Literal["development", "production", "test"] = "development"
    aetherforge_host: str = "127.0.0.1"
    aetherforge_port: int = Field(default=8765, ge=1024, le=65535)
    aetherforge_log_level: Literal["debug", "info", "warning", "error"] = "info"

    # ── BitNet Model ──────────────────────────────────────────────
    # microsoft/bitnet-b1.58-2b-4t is the default model.
    # n_gpu_layers=-1 → offload all layers to Metal GPU on M1.
    # n_ctx=4096 → context window: 4k tokens, balanced speed/quality.
    bitnet_model_path: Path = Path("./models/bitnet-b1.58-2b-4t.gguf")
    bitnet_n_ctx: int = Field(default=4096, ge=512, le=32768)
    bitnet_n_gpu_layers: int = Field(default=-1, ge=-1)  # -1 = all
    bitnet_n_threads: int = Field(default=8, ge=1, le=64)
    bitnet_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    bitnet_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    bitnet_max_tokens: int = Field(default=1024, ge=64, le=8192)

    # ── Silicon Colosseum / Guardrails ────────────────────────────
    # OPA can run embedded (subprocess) or as a separate server.
    opa_mode: Literal["embedded", "server"] = "embedded"
    opa_server_url: str = "http://localhost:8181"
    # Maximum tool calls per agent turn (prevents runaway loops)
    silicon_colosseum_max_tool_calls: int = Field(default=8, ge=1, le=50)
    # Minimum faithfulness score (0.0–1.0) to allow output
    silicon_colosseum_min_faithfulness: float = Field(default=0.92, ge=0.0, le=1.0)

    # ── Storage ───────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    replay_buffer_path: Path = Path("./data/replay_buffer.parquet")
    sqlcipher_key_file: Path = Path("./data/.db_key")
    chroma_path: Path = Path("./data/chroma")

    # ── Telemetry (local Langfuse) ────────────────────────────────
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = "af_local"
    langfuse_secret_key: str = "af_local_secret"

    # ── Neo4j / X-Ray Causal Graph ────────────────────────────────
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "aetherforge123"

    # ── Nightly OPLoRA Scheduler ──────────────────────────────────
    # Fires at 3:00 AM local time if battery > 30%
    oploра_nightly_hour: int = Field(default=3, ge=0, le=23)
    oploра_nightly_minute: int = Field(default=0, ge=0, le=59)
    oploра_min_battery_pct: int = Field(default=30, ge=0, le=100)

    # ── OPLoRA Hyperparameters ────────────────────────────────────
    # rank_k: number of singular vectors to preserve from past tasks.
    # A higher rank_k → more past knowledge preserved, but less
    # capacity for new knowledge. 64 is a good default for 2B models.
    oploра_rank_k: int = Field(default=64, ge=4, le=512)
    oploра_lora_r: int = Field(default=16, ge=1, le=256)
    oploра_lora_alpha: float = Field(default=32.0, ge=1.0)
    oploра_lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    oploра_learning_rate: float = Field(default=1e-4, gt=0.0)
    oploра_epochs: int = Field(default=3, ge=1, le=50)

    # ── Validators ────────────────────────────────────────────────
    @field_validator("bitnet_model_path", "data_dir", "chroma_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand ~ and relative paths at validation time."""
        return Path(os.path.expandvars(str(v))).expanduser()

    @computed_field  # type: ignore[misc]
    @property
    def is_production(self) -> bool:
        return self.aetherforge_env == "production"

    @computed_field  # type: ignore[misc]
    @property
    def api_base_url(self) -> str:
        return f"http://{self.aetherforge_host}:{self.aetherforge_port}"

    def ensure_data_dirs(self) -> None:
        """Create required data directories if they don't exist."""
        for p in [
            self.data_dir,
            self.chroma_path,
            self.data_dir / "replay",
            self.data_dir / "sessions",
            self.data_dir / "logs",
            self.data_dir / "lora_checkpoints",
        ]:
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AetherForgeSettings:
    """
    Return the singleton settings object.
    Cached after first call — safe to call from anywhere.
    """
    settings = AetherForgeSettings()
    settings.ensure_data_dirs()
    return settings
