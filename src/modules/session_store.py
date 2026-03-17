# AetherForge v1.0 — src/modules/session_store.py
# ─────────────────────────────────────────────────────────────────
# Persistent chat session storage backed by SQLite.
#
# Replaces MetaAgent._session_memories (ephemeral in-process dict)
# with a durable store that survives server restarts.
#
# Schema:
#   sessions  (id, module, title, created_at, updated_at)
#   messages  (id, session_id, role, content, ts, metadata_json)
#
# Thread safety: write operations use a threading.Lock.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json

import structlog  # type: ignore[import-untyped]

try:
    import sqlcipher3 as sqlite3  # type: ignore[import-untyped]

    HAS_SQLCIPHER = True
except ImportError:
    import sqlite3

    HAS_SQLCIPHER = False
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = structlog.get_logger("aetherforge.session_store")

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    module      TEXT NOT NULL DEFAULT 'localbuddy',
    title       TEXT NOT NULL DEFAULT 'New Session',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
    content     TEXT NOT NULL,
    ts          REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_sessions_module  ON sessions(module, updated_at DESC);
"""


@dataclass
class SessionSummary:
    id: str
    module: str
    title: str
    created_at: float
    updated_at: float
    message_count: int = 0


@dataclass
class StoredMessage:
    id: str
    session_id: str
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    ts: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    """
    Durable SQLite-backed store for chat sessions.

    Lifecycle:
        store = SessionStore(db_path)  ← created once in AppState
        store.create_session(module)   ← called by chat endpoint
        store.append_message(...)      ← called after every LLM turn
        store.get_messages(session_id) ← called on page load / restart
    """

    def __init__(
        self, db_path: str | Path = "./data/sessions.db", key_file: str | Path | None = None
    ) -> None:
        self.db_path = Path(db_path)
        self.key_file = Path(key_file) if key_file else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()
        self._init_schema()
        logger.info("SessionStore initialized at %s (SQLCipher=%s)", self.db_path, HAS_SQLCIPHER)

    # ── Connection ────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn = conn
            kf = self.key_file
            if HAS_SQLCIPHER and kf and kf.exists():
                try:
                    key = kf.read_text().strip()
                    # SQLCipher needs the key immediately after opening
                    conn.execute(f"PRAGMA key = '{key}'")
                    # Verify key by trying a simple operation
                    conn.execute("SELECT count(*) FROM sqlite_master")
                    logger.debug("SessionStore: SECURE (SQLCipher active)")
                except Exception as e:
                    err_msg = str(e).lower()
                    if "file is not a database" in err_msg or "not authenticated" in err_msg:
                        logger.warning(
                            "SessionStore: DB is plain SQLite or key mismatch. Falling back to plain mode."
                        )
                        # RESET: We MUST close and re-open to clear the "poisoned" encrypted state
                        # which can otherwise lead to a MemoryError in some sqlcipher3 builds.
                        conn.close()
                        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                        self._conn = conn
                    else:
                        logger.error("SessionStore: Encryption key error: %s", e)

            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
        
        assert self._conn is not None
        return self._conn


    def _init_schema(self) -> None:
        with self._write_lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA)
            conn.commit()

    # ── Session CRUD ──────────────────────────────────────────────

    def create_session(self, module: str = "localbuddy", session_id: str | None = None) -> str:
        """Create a new session and return its ID."""
        sid = session_id or str(uuid.uuid4())
        now = time.time()
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, module, title, created_at, updated_at) "
                "VALUES (?, ?, 'New Session', ?, ?)",
                (sid, module, now, now),
            )
            conn.commit()
        return sid

    def rename_session(self, session_id: str, title: str) -> None:
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
                (title[:120], time.time(), session_id),  # type: ignore[misc]
            )
            conn.commit()

    def delete_session(self, session_id: str) -> None:
        """Delete session and all its messages (CASCADE)."""
        with self._write_lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            conn.commit()
        logger.info("Session %s deleted", session_id)

    def list_sessions(self, module: str | None = None) -> list[SessionSummary]:
        conn = self._get_conn()
        if module:
            rows = conn.execute(
                "SELECT s.id, s.module, s.title, s.created_at, s.updated_at, "
                "COUNT(m.id) as message_count "
                "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
                "WHERE s.module=? GROUP BY s.id ORDER BY s.updated_at DESC",
                (module,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT s.id, s.module, s.title, s.created_at, s.updated_at, "
                "COUNT(m.id) as message_count "
                "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
                "GROUP BY s.id ORDER BY s.updated_at DESC"
            ).fetchall()
        return [
            SessionSummary(
                id=r["id"],
                module=r["module"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["message_count"],
            )
            for r in rows
        ]

    # ── Message CRUD ──────────────────────────────────────────────

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Append a message to a session. Auto-titles new sessions from first user message."""
        msg_id = str(uuid.uuid4())
        now = time.time()

        with self._write_lock:
            conn = self._get_conn()

            # Auto-title: if this is the first user message, set session title
            if role == "user":
                existing = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user'",
                    (session_id,),
                ).fetchone()[0]
                if existing == 0:
                    title = content[:80].replace("\n", " ").strip()  # type: ignore[misc]
                    conn.execute(
                        "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
                        (title, now, session_id),
                    )

            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, ts, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, session_id, role, content, now, json.dumps(metadata or {})),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (now, session_id),
            )
            conn.commit()

        return msg_id

    def append_turn(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
        assistant_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Persist the user/assistant pair atomically under one lock."""
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        now = time.time()

        with self._write_lock:
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user'",
                (session_id,),
            ).fetchone()[0]
            if existing == 0:
                title = user_content[:80].replace("\n", " ").strip()  # type: ignore[misc]
                conn.execute(
                    "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
                    (title, now, session_id),
                )

            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, ts, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, session_id, "user", user_content, now, "{}"),
            )
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, ts, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    assistant_id,
                    session_id,
                    "assistant",
                    assistant_content,
                    now + 0.0001,
                    json.dumps(assistant_metadata or {}),
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (now + 0.0001, session_id),
            )
            conn.commit()
        return user_id, assistant_id

    def get_messages(self, session_id: str) -> list[StoredMessage]:
        """Return messages for a session ordered by timestamp."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, session_id, role, content, ts, metadata FROM messages "
            "WHERE session_id=? ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
        return [
            StoredMessage(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                ts=r["ts"],
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for r in rows
        ]

    def session_exists(self, session_id: str) -> bool:
        conn = self._get_conn()
        return bool(conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone())

    # ── LangChain message conversion ─────────────────────────────

    def to_langchain_messages(self, session_id: str) -> list[Any]:
        """
        Convert stored messages to LangChain message objects for MetaAgent.
        Returns a list compatible with _session_memories values.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # type: ignore[import-untyped]

        from src.meta_agent import _SYSTEM_PROMPT  # type: ignore[import-untyped]

        msgs = self.get_messages(session_id)
        lc_msgs: list[Any] = [SystemMessage(content=_SYSTEM_PROMPT)]
        for m in msgs:
            if m.role == "user":
                lc_msgs.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                # Supply ONLY the clean, user-visible answer text back into the LLM context.
                # If we supply the raw .content (with <think> and JSON), the LLM gets trapped
                # in a context bleed loop, echoing old tool patterns instead of answering fresh queries.
                clean_content = m.metadata.get("answer_text") or m.content
                lc_msgs.append(AIMessage(content=clean_content.strip()))
            # skip stored 'system' rows (already injected above)
        return lc_msgs

    def close(self) -> None:
        conn = self._conn
        if conn:
            conn.close()
            self._conn = None
