from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
    document_id         TEXT PRIMARY KEY,
    source              TEXT NOT NULL UNIQUE,
    file_type           TEXT NOT NULL,
    ingest_status       TEXT NOT NULL,
    parser              TEXT NOT NULL DEFAULT 'unknown',
    chunk_count         INTEGER NOT NULL DEFAULT 0,
    image_pages_pending INTEGER NOT NULL DEFAULT 0,
    last_error          TEXT,
    selected            INTEGER NOT NULL DEFAULT 1,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(ingest_status);
"""


@dataclass
class DocumentRecord:
    document_id: str
    source: str
    file_type: str
    ingest_status: str
    parser: str
    chunk_count: int
    image_pages_pending: int
    last_error: str | None
    selected: bool
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected"] = bool(self.selected)
        return payload


class DocumentRegistry:
    def __init__(self, db_path: str | Path = "./data/document_registry.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        assert self._conn is not None
        return self._conn

    def _ensure_schema(self) -> None:
        with self._write_lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA)
            conn.commit()

    def upsert_document(
        self,
        *,
        source: str,
        file_type: str,
        ingest_status: str,
        parser: str = "unknown",
        chunk_count: int = 0,
        image_pages_pending: int = 0,
        last_error: str | None = None,
        selected: bool | None = None,
    ) -> DocumentRecord:
        now = time.time()
        with self._write_lock:
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT document_id, created_at, selected FROM documents WHERE source = ?",
                (source,),
            ).fetchone()
            if existing:
                document_id = str(existing["document_id"])
                created_at = float(existing["created_at"])
                selected_value = int(existing["selected"]) if existing["selected"] is not None else 1
                conn.execute(
                    """
                    UPDATE documents
                    SET file_type = ?, ingest_status = ?, parser = ?, chunk_count = ?,
                        image_pages_pending = ?, last_error = ?, selected = ?, updated_at = ?
                    WHERE source = ?
                    """,
                    (
                        file_type,
                        ingest_status,
                        parser,
                        int(chunk_count),
                        int(image_pages_pending),
                        last_error,
                        selected_value if selected is None else (1 if selected else 0),
                        now,
                        source,
                    ),
                )
            else:
                document_id = str(uuid.uuid4())
                created_at = now
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, source, file_type, ingest_status, parser, chunk_count,
                        image_pages_pending, last_error, selected, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        source,
                        file_type,
                        ingest_status,
                        parser,
                        int(chunk_count),
                        int(image_pages_pending),
                        last_error,
                        1 if selected is not False else 0,
                        created_at,
                        now,
                    ),
                )
            conn.commit()
        record = self.get_by_source(source)
        assert record is not None
        return record

    def update_document(self, document_id: str, **fields: Any) -> DocumentRecord | None:
        if not fields:
            return self.get_by_id(document_id)
        allowed = {
            "source",
            "file_type",
            "ingest_status",
            "parser",
            "chunk_count",
            "image_pages_pending",
            "last_error",
            "selected",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_by_id(document_id)

        assignments = []
        params: list[Any] = []
        for key, value in updates.items():
            assignments.append(f"{key} = ?")
            if key == "selected":
                params.append(1 if bool(value) else 0)
            else:
                params.append(value)
        assignments.append("updated_at = ?")
        params.append(time.time())
        params.append(document_id)

        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                f"UPDATE documents SET {', '.join(assignments)} WHERE document_id = ?",
                tuple(params),
            )
            conn.commit()
        return self.get_by_id(document_id)

    def get_by_source(self, source: str) -> DocumentRecord | None:
        row = self._get_conn().execute(
            "SELECT * FROM documents WHERE source = ?",
            (source,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_by_id(self, document_id: str) -> DocumentRecord | None:
        row = self._get_conn().execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        rows = self._get_conn().execute(
            "SELECT * FROM documents ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count_documents(self) -> int:
        row = self._get_conn().execute("SELECT COUNT(*) AS total FROM documents").fetchone()
        return int(row["total"]) if row else 0

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=str(row["document_id"]),
            source=str(row["source"]),
            file_type=str(row["file_type"]),
            ingest_status=str(row["ingest_status"]),
            parser=str(row["parser"] or "unknown"),
            chunk_count=int(row["chunk_count"] or 0),
            image_pages_pending=int(row["image_pages_pending"] or 0),
            last_error=str(row["last_error"]) if row["last_error"] else None,
            selected=bool(row["selected"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def close(self) -> None:
        conn = self._conn
        if conn is not None:
            conn.close()
            self._conn = None
