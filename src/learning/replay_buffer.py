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
import time
import uuid
from typing import Any

import structlog

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

from src.config import AetherForgeSettings

logger = structlog.get_logger("aetherforge.replay_buffer")

# ── Parquet schema ────────────────────────────────────────────────
# Explicit schema ensures consistent types across all writes.
if pa:
    _SCHEMA = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("module", pa.string()),
            pa.field("prompt", pa.binary()),  # Encrypted
            pa.field("response", pa.binary()),  # Encrypted
            pa.field("faithfulness_score", pa.float32()),
            pa.field("tool_calls_json", pa.string()),
            pa.field("timestamp_utc", pa.float64()),
            pa.field("is_used_for_training", pa.bool_()),
            pa.field("novelty_score", pa.float32()),
        ]
    )
else:
    _SCHEMA = None


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
        # Partitioned dataset directory
        self._root_path = settings.data_dir / "replay_dataset"
        self._path = self._root_path  # For backward compatibility in logs
        self._key_file = settings.sqlcipher_key_file
        self._fernet: Fernet | None = None
        self._write_lock = asyncio.Lock()
        self._pending: list[dict[str, Any]] = []  # In-memory queue
        self._flush_threshold = 10  # Flush every N records

    async def initialize(self) -> None:
        """Load or generate encryption key and ensure Parquet file exists."""
        if not Fernet:
            logger.warning("cryptography not installed - skipping encryption key loading")
        else:
            self._fernet = await asyncio.get_event_loop().run_in_executor(
                None, self._load_or_create_key
            )

        self._root_path.mkdir(parents=True, exist_ok=True)
        if not pa or not pq:
            logger.warning("pyarrow not installed - skipping replay buffer initialization")
            return

        # Check if dataset is empty or has data
        if not any(self._root_path.iterdir()):
            logger.info("Created new replay buffer dataset directory at %s", self._root_path)
        else:
            stats = await self.get_stats()
            logger.info(
                "Replay buffer loaded: %d records across %d modules at %s",
                stats["total_records"],
                len(stats["modules"]),
                self._root_path,
            )

    def _load_or_create_key(self) -> Fernet:
        """
        Load Fernet key from disk, or generate a new one.
        """
        if not Fernet:
            return None

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
            "prompt": prompt[:4096],  # Truncate long prompts
            "response": response[:8192],  # Truncate long responses
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
        await asyncio.get_event_loop().run_in_executor(None, self._append_to_parquet, records)
        logger.debug("Flushed %d records to replay buffer", len(records))

    def _append_to_parquet(self, records: list[dict[str, Any]]) -> None:
        """
        Append records to the partitioned Parquet dataset.
        Encrypts prompts/responses before storage.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        encrypted_records = []
        for r in records:
            enc_r = r.copy()
            if self._fernet:
                enc_r["prompt"] = self._encrypt(r["prompt"].encode())
                enc_r["response"] = self._encrypt(r["response"].encode())
            else:
                enc_r["prompt"] = r["prompt"].encode()
                enc_r["response"] = r["response"].encode()
            encrypted_records.append(enc_r)

        table = pa.Table.from_pylist(encrypted_records, schema=_SCHEMA)

        # Write to partitioned dataset (module-based folders)
        pq.write_to_dataset(
            table,
            root_path=str(self._root_path),
            partition_cols=["module"],
            existing_data_behavior="overwrite_or_ignore",
        )

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
        module: str | None = None,
        min_faithfulness: float = 0.0,
        exclude_used: bool = False,
    ) -> list[dict[str, Any]]:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        if not self._root_path.exists() or not any(self._root_path.iterdir()):
            return []

        try:
            dataset = pq.ParquetDataset(str(self._root_path))
            table = dataset.read()

            # Filters (Using operator overloading for compatibility)
            mask = (pc.field("faithfulness_score") >= min_faithfulness)
            if module:
                mask = mask & (pc.field("module") == module)
            if exclude_used:
                mask = mask & (pc.field("is_used_for_training") == False)

            filtered_table = table.filter(mask)

            # Sample (or take last N)
            rows = filtered_table.to_pylist()
            sampled = rows[-n:] if len(rows) > n else rows

            # Decrypt
            for r in sampled:
                if self._fernet:
                    try:
                        r["prompt"] = self._decrypt(r["prompt"]).decode()
                        r["response"] = self._decrypt(r["response"]).decode()
                    except Exception:
                        r["prompt"] = "[Decryption Error]"
                        r["response"] = "[Decryption Error]"
                else:
                    r["prompt"] = r["prompt"].decode()
                    r["response"] = r["response"].decode()

            return sampled
        except Exception as e:
            logger.error("Failed to read sample from replay buffer: %s", e)
            return []

    async def get_stats(self) -> dict[str, Any]:
        """Return buffer statistics for the TuneLab dashboard."""

        def _stats() -> dict[str, Any]:
            if not self._root_path.exists() or not any(self._root_path.iterdir()):
                return {"total_records": 0, "size_mb": 0.0, "modules": {}}
            try:
                # Use dataset to read all partitions
                import pyarrow.parquet as pq

                dataset = pq.ParquetDataset(str(self._root_path))
                table = dataset.read()

                total_records = len(table)
                # Size calculation: sum of all parquet files in the dataset
                total_size_bytes = sum(f.stat().st_size for f in self._root_path.rglob("*.parquet"))

                # Modules breakdown
                # Note: 'module' is a transition column in partitioned datasets
                # but pyarrow handles it correctly in the resulting table.
                modules_col = table.column("module").to_pylist()
                from collections import Counter

                modules_stats = dict(Counter(modules_col))

                return {
                    "total_records": total_records,
                    "size_mb": float(round(total_size_bytes / (1024 * 1024), 2)),
                    "modules": modules_stats,
                }
            except Exception as e:
                logger.error("Failed to get replay buffer stats: %s", e)
                return {"total_records": 0, "size_mb": 0.0, "modules": {}}

        return await asyncio.get_event_loop().run_in_executor(None, _stats)

    async def mark_as_used(self, record_ids: list[str]) -> int:
        """Mark records as used for training. Returns count updated."""

        def _mark() -> int:
            if not self._root_path.exists():
                return 0
            # For partitioned datasets, we read the whole table, modify, and overwrite.
            # In a production system, we'd use an incremental update or separate metadata DB.
            dataset = pq.ParquetDataset(str(self._root_path))
            table = dataset.read()
            df = table.to_pandas()
            mask = df["id"].isin(record_ids)
            df.loc[mask, "is_used_for_training"] = True
            
            # Write back as a new single file in the root for now to simplify marking,
            # or overwrite the dataset. Overwriting a partitioned dataset is complex.
            # We'll write to a "marked" partition or simplify the update logic.
            # Simplified for now: overwrite the whole table back to the partitioned structure.
            from pyarrow import Table
            new_table = Table.from_pandas(df, schema=_SCHEMA)
            pq.write_to_dataset(
                new_table,
                root_path=str(self._root_path),
                partition_cols=["module"],
                existing_data_behavior="overwrite_or_ignore",
            )
            return int(mask.sum())

        return await asyncio.get_event_loop().run_in_executor(None, _mark)
