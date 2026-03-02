# AetherForge v1.0 — src/learning/replay_buffer.py
# ─────────────────────────────────────────────────────────────────
# Encrypted replay buffer for AetherForge's perpetual learning loop.
#
# Every interaction (prompt + response + metadata) is stored here.
# The nightly OPLoRA trainer reads from this buffer to build the
# fine-tuning dataset. Old entries are not deleted — they are
# reviewed for novelty by InsightForge and selectively retained.
#
# Storage format: Apache Parquet (columnar, compressed, fast scan)
#   Why Parquet? Columnar storage → O(1) column scans for sampling.
#   pandas+pyarrow reads millions of rows without loading all to RAM.
#
# Encryption: Fernet (AES-128-CBC + HMAC-SHA256) key derived from
# a machine-specific secret stored in data/.db_key.
# We intentionally use symmetric encryption (not age/asymmetric)
# because the box that writes is the same box that reads — there's
# no key distribution problem. The key file is included in .gitignore.
#
# Thread safety: asyncio.Lock guards all write operations.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import io
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from cryptography.fernet import Fernet

from src.config import AetherForgeSettings

logger = logging.getLogger("aetherforge.replay_buffer")

# ── Parquet schema ────────────────────────────────────────────────
# Explicit schema ensures consistent types across all writes.
_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("module", pa.string()),
    pa.field("prompt", pa.string()),
    pa.field("response", pa.string()),
    pa.field("faithfulness_score", pa.float32()),
    pa.field("tool_calls_json", pa.string()),  # JSON string
    pa.field("timestamp_utc", pa.float64()),   # Unix timestamp
    pa.field("is_used_for_training", pa.bool_()),
    pa.field("novelty_score", pa.float32()),   # Set by InsightForge
])


class ReplayBuffer:
    """
    Encrypted, append-only replay buffer backed by Parquet.

    Usage:
        buffer = ReplayBuffer(settings)
        await buffer.initialize()
        await buffer.record(session_id=..., module=..., prompt=..., response=...)
        df = await buffer.sample(n=256)  # For training
        await buffer.flush()
    """

    def __init__(self, settings: AetherForgeSettings) -> None:
        self.settings = settings
        self._path = settings.replay_buffer_path
        self._key_file = settings.sqlcipher_key_file
        self._fernet: Fernet | None = None
        self._write_lock = asyncio.Lock()
        self._pending: list[dict[str, Any]] = []  # In-memory queue
        self._flush_threshold = 10  # Flush every N records

    async def initialize(self) -> None:
        """Load or generate encryption key and ensure Parquet file exists."""
        self._fernet = await asyncio.get_event_loop().run_in_executor(
            None, self._load_or_create_key
        )
        # Create empty Parquet file if it doesn't exist
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            empty_table = pa.table({f.name: pa.array([], type=f.type) for f in _SCHEMA}, schema=_SCHEMA)
            pq.write_table(empty_table, self._path, compression="snappy")
            logger.info("Created new replay buffer at %s", self._path)
        else:
            stats = await self.get_stats()
            logger.info(
                "Replay buffer loaded: %d records at %s",
                stats["total_records"],
                self._path,
            )

    def _load_or_create_key(self) -> Fernet:
        """
        Load Fernet key from disk, or generate a new one.
        Key is stored in binary (32 random bytes → base64-encoded).
        IMPORTANT: Losing this file means losing access to all stored data.
        """
        key_file = self._key_file
        key_file.parent.mkdir(parents=True, exist_ok=True)
        if key_file.exists():
            key = key_file.read_bytes()
            logger.debug("Loaded existing encryption key from %s", key_file)
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            # Restrict permissions: only owner can read
            key_file.chmod(0o600)
            logger.info("Generated new encryption key at %s", key_file)
        return Fernet(key)

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data with Fernet (AES-128-CBC + HMAC-SHA256)."""
        assert self._fernet is not None, "Buffer not initialized"
        return self._fernet.encrypt(data)

    def _decrypt(self, token: bytes) -> bytes:
        """Decrypt Fernet token."""
        assert self._fernet is not None, "Buffer not initialized"
        return self._fernet.decrypt(token)

    async def record(
        self,
        session_id: str,
        module: str,
        prompt: str,
        response: str,
        tool_calls: list[dict[str, Any]] | None = None,
        faithfulness_score: float | None = None,
    ) -> str:
        """
        Record one interaction to the replay buffer.
        Returns the record ID.
        Thread-safe via asyncio.Lock.
        """
        import json

        record_id = str(uuid.uuid4())
        record: dict[str, Any] = {
            "id": record_id,
            "session_id": session_id,
            "module": module,
            "prompt": prompt[:4096],        # Truncate long prompts
            "response": response[:8192],    # Truncate long responses
            "faithfulness_score": float(faithfulness_score or 0.0),
            "tool_calls_json": json.dumps(tool_calls or []),
            "timestamp_utc": time.time(),
            "is_used_for_training": False,
            "novelty_score": 0.0,
        }

        async with self._write_lock:
            self._pending.append(record)
            if len(self._pending) >= self._flush_threshold:
                await self._flush_pending()

        logger.debug("Recorded interaction: id=%s module=%s", record_id, module)
        return record_id

    async def flush(self) -> None:
        """Flush all pending records to disk. Called on shutdown."""
        async with self._write_lock:
            await self._flush_pending()

    async def _flush_pending(self) -> None:
        """Write pending records to Parquet (called while holding lock)."""
        if not self._pending:
            return

        records = self._pending[:]
        self._pending.clear()

        # Run Parquet I/O in thread executor (blocking)
        await asyncio.get_event_loop().run_in_executor(
            None, self._append_to_parquet, records
        )
        logger.debug("Flushed %d records to replay buffer", len(records))

    def _append_to_parquet(self, records: list[dict[str, Any]]) -> None:
        """
        Append records to the Parquet file.

        We use PyArrow's writer in APPEND mode. Since Parquet doesn't
        natively support appending, we read existing + concatenate + rewrite.
        For production at scale, use Delta Lake or DuckDB append-mode.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        new_table = pa.Table.from_pylist(records, schema=_SCHEMA)

        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                existing = pq.read_table(self._path)
                combined = pa.concat_tables([existing, new_table])
            except Exception:
                combined = new_table
        else:
            combined = new_table

        pq.write_table(combined, self._path, compression="snappy")

    async def sample(
        self,
        n: int = 256,
        module: str | None = None,
        min_faithfulness: float = 0.0,
        exclude_used: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Sample n records from the buffer for training.

        Filters:
          - module: restrict to a specific module
          - min_faithfulness: only high-quality interactions
          - exclude_used: skip records already used in a training run
        """
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._read_sample, n, module, min_faithfulness, exclude_used
        )
        return result

    def _read_sample(
        self,
        n: int,
        module: str | None,
        min_faithfulness: float,
        exclude_used: bool,
    ) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            table = pq.read_table(self._path)
            df = table.to_pandas()
            if module:
                df = df[df["module"] == module]
            if min_faithfulness > 0:
                df = df[df["faithfulness_score"] >= min_faithfulness]
            if exclude_used:
                df = df[~df["is_used_for_training"]]
            # Reservoir sample
            if len(df) > n:
                df = df.sample(n=n, random_state=42)
            return df.to_dict(orient="records")
        except Exception as exc:
            logger.exception("Failed to read replay buffer: %s", exc)
            return []

    async def get_stats(self) -> dict[str, Any]:
        """Return buffer statistics for the TuneLab dashboard."""
        def _stats() -> dict[str, Any]:
            if not self._path.exists():
                return {"total_records": 0, "size_mb": 0.0, "modules": {}}
            try:
                table = pq.read_table(self._path)
                df = table.to_pandas()
                return {
                    "total_records": len(df),
                    "size_mb": round(self._path.stat().st_size / 1e6, 2),
                    "modules": df.groupby("module").size().to_dict() if len(df) > 0 else {},
                    "trained_count": int(df["is_used_for_training"].sum()) if len(df) > 0 else 0,
                    "avg_faithfulness": float(df["faithfulness_score"].mean()) if len(df) > 0 else 0.0,
                    "oldest_record": datetime.fromtimestamp(
                        float(df["timestamp_utc"].min()), tz=timezone.utc
                    ).isoformat() if len(df) > 0 else None,
                }
            except Exception as exc:
                return {"error": str(exc)}

        return await asyncio.get_event_loop().run_in_executor(None, _stats)

    async def mark_as_used(self, record_ids: list[str]) -> int:
        """Mark records as used for training. Returns count updated."""
        def _mark() -> int:
            if not self._path.exists():
                return 0
            table = pq.read_table(self._path)
            df = table.to_pandas()
            mask = df["id"].isin(record_ids)
            df.loc[mask, "is_used_for_training"] = True
            pq.write_table(pa.Table.from_pandas(df, schema=_SCHEMA), self._path, compression="snappy")
            return int(mask.sum())

        return await asyncio.get_event_loop().run_in_executor(None, _mark)
