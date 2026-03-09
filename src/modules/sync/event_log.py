import json
import structlog
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

logger = structlog.get_logger("aetherforge.sync.event_log")

class HybridLogicalClock:
    """
    A simple Hybrid Logical Clock (HLC) for CRDT causal ordering.
    Format: <physical_time_ms>-<logical_counter>-<node_id>
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.physical_time = 0
        self.logical_counter = 0

    def now(self) -> str:
        current_time = int(time.time() * 1000)
        if current_time > self.physical_time:
            self.physical_time = current_time
            self.logical_counter = 0
        else:
            self.logical_counter += 1
        
        # Zero-pad logical counter for lexicographical sorting
        return f"{self.physical_time}-{self.logical_counter:04d}-{self.node_id}"

class EventLog:
    """
    SQLite-backed Write-Ahead Log (WAL) for Eventual Consistency.
    Every state change in AetherForge is written here as an immutable event.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for concurrent reads/writes
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_events (
                    id TEXT PRIMARY KEY,
                    hlc_timestamp TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    origin_device_id TEXT NOT NULL,
                    encrypted_blob BLOB,
                    is_synced INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hlc on sync_events(hlc_timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity on sync_events(entity_id);")

    def append_event(self, hlc_timestamp: str, entity_type: str, entity_id: str, action: str, origin_device_id: str, encrypted_blob: bytes) -> str:
        """
        Append a new encrypted state mutation to the WAL.
        """
        event_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sync_events (id, hlc_timestamp, entity_type, entity_id, action, origin_device_id, encrypted_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (event_id, hlc_timestamp, entity_type, entity_id, action, origin_device_id, sqlite3.Binary(encrypted_blob)))
            conn.commit()
        logger.debug(f"Appended event {event_id} for {entity_type}:{entity_id} to WAL")
        return event_id

    def get_events_since(self, since_hlc: str, limit: int = 500) -> list[dict[str, Any]]:
        """
        Fetch a batch of events that occurred after the specified HLC timestamp.
        Used when a peer requests synchronization.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, hlc_timestamp, entity_type, entity_id, action, origin_device_id, encrypted_blob
                FROM sync_events
                WHERE hlc_timestamp > ?
                ORDER BY hlc_timestamp ASC
                LIMIT ?
            """, (since_hlc, limit))
            
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def merge_foreign_event(self, event_dict: dict[str, Any]) -> bool:
        """
        Idempotent merge: Attempt to insert an event received from a peer.
        If it already exists, ignore it (Conflict-Free Replicated Data Type property).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO sync_events (id, hlc_timestamp, entity_type, entity_id, action, origin_device_id, encrypted_blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_dict["id"],
                    event_dict["hlc_timestamp"],
                    event_dict["entity_type"],
                    event_dict["entity_id"],
                    event_dict["action"],
                    event_dict["origin_device_id"],
                    sqlite3.Binary(event_dict["encrypted_blob"])
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Event already merged (UNIQUE constraint failed)
            return False
