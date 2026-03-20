from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog

from src.modules.document_registry import DocumentRecord, DocumentRegistry
from src.modules.ragforge.vlm_enrich import async_vlm_enrich
from src.modules.ragforge_indexer import index_document
from src.modules.streamsync.graph import emit_event
from src.utils import safe_create_task

logger = structlog.get_logger("aetherforge.document_intelligence")


class DocumentIntelligenceService:
    def __init__(
        self,
        *,
        settings: Any,
        vector_store: Any,
        sparse_index: Any,
        document_registry: DocumentRegistry,
        selected_vlm_id_getter: Any,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.document_registry = document_registry
        self._selected_vlm_id_getter = selected_vlm_id_getter

    async def ingest_path(self, file_path: Path) -> dict[str, Any]:
        source = file_path.name
        file_type = file_path.suffix.lower().lstrip(".") or "unknown"
        record = self.document_registry.upsert_document(
            source=source,
            file_type=file_type,
            ingest_status="extracting_text",
            parser="pending",
            chunk_count=0,
            image_pages_pending=0,
            last_error=None,
            selected=None,
        )

        try:
            result = await asyncio.to_thread(
                index_document,
                file_path,
                self.vector_store,
                self.sparse_index,
                self.document_registry,   # enables boot-sweep idempotency guard
            )
        except Exception as exc:
            logger.exception("Document ingest failed", source=source, error=str(exc))
            failed = self.document_registry.update_document(
                record.document_id,
                ingest_status="failed",
                last_error=str(exc),
                parser="failed",
                chunk_count=0,
                image_pages_pending=0,
            )
            emit_event(
                event_type="document_index_failed",
                source="DocumentIntelligence",
                payload={"filename": source, "error": str(exc)},
            )
            return self._build_response(failed or record)

        parser = str(result.get("parser", "unknown"))
        chunks_added = int(result.get("chunks_added", 0) or 0)
        image_pages = list(result.get("image_pages", []) or [])
        image_pages_pending = len(image_pages)
        last_error = result.get("error")

        ingest_status = "ready"
        if last_error:
            ingest_status = "failed"
        elif image_pages_pending and chunks_added > 0:
            ingest_status = "partial"
        elif image_pages_pending:
            ingest_status = "ocr_pending"
        elif chunks_added == 0:
            ingest_status = "failed"
            last_error = "No retrievable chunks were extracted from the document."

        updated = self.document_registry.update_document(
            record.document_id,
            ingest_status=ingest_status,
            parser=parser,
            chunk_count=chunks_added,
            image_pages_pending=image_pages_pending,
            last_indexed_mtime=result.get("last_indexed_mtime"),
            last_error=last_error,
        )
        assert updated is not None

        if image_pages_pending:
            safe_create_task(
                self._run_vlm_enrichment(file_path, updated.document_id, image_pages),
                name=f"vlm_enrich_{source}",
            )

        emit_event(
            event_type="document_indexed",
            source="DocumentIntelligence",
            payload={
                "filename": source,
                "chunks": chunks_added,
                "status": ingest_status,
                "image_pages_pending": image_pages_pending,
            },
        )
        return self._build_response(updated)

    async def _run_vlm_enrichment(
        self,
        file_path: Path,
        document_id: str,
        image_pages: list[int],
    ) -> None:
        current = self.document_registry.get_by_id(document_id)
        if current is None:
            return

        self.document_registry.update_document(
            document_id,
            ingest_status="ocr_running",
            last_error=None,
            image_pages_pending=len(image_pages),
        )

        try:
            result = await async_vlm_enrich(
                file_path=file_path,
                image_pages=image_pages,
                vector_store=self.vector_store,
                vlm_id=str(self._selected_vlm_id_getter() or "smolvlm-256m"),
                sparse_index=self.sparse_index,
            )
            if not isinstance(result, dict):
                # Fallback for unexpected return types
                result = {"status": "failed", "chunks_added": 0}

            if result.get("status") == "skipped_memory":
                # Memory ceiling hit — set back to pending for retry
                self.document_registry.update_document(
                    document_id,
                    ingest_status="ocr_pending",
                    last_error="Deferred: system memory too high for VLM processing.",
                )
                emit_event(
                    event_type="document_ocr_deferred",
                    source="DocumentIntelligence",
                    payload={"filename": file_path.name, "reason": "memory_limit"},
                )
                return

            vlm_chunks = int(result.get("chunks_added", 0) or 0)
            record = self.document_registry.get_by_id(document_id)
            if record is None:
                return

            total_chunks = int(record.chunk_count) + vlm_chunks
            last_error = result.get("last_error")
            if total_chunks > 0 and not last_error:
                status = "ready"
            elif record.chunk_count > 0:
                status = "partial"
                if not last_error:
                    last_error = "OCR completed without adding visual chunks."
            else:
                status = "failed"
                if not last_error:
                    last_error = "OCR completed without extracting searchable content."

            self.document_registry.update_document(
                document_id,
                ingest_status=status,
                parser=f"{record.parser}+vlm",
                chunk_count=total_chunks,
                image_pages_pending=0,
                last_error=last_error,
            )
            emit_event(
                event_type="document_ocr_completed",
                source="DocumentIntelligence",
                payload={
                    "filename": file_path.name,
                    "status": status,
                    "chunks_added": vlm_chunks,
                    "vlm_id": result.get("vlm_id"),
                },
            )
        except Exception as exc:
            logger.exception("VLM enrichment failed", source=file_path.name, error=str(exc))
            record = self.document_registry.get_by_id(document_id)
            fallback_status = "partial" if record and record.chunk_count > 0 else "failed"
            self.document_registry.update_document(
                document_id,
                ingest_status=fallback_status,
                image_pages_pending=0,
                last_error=str(exc),
            )
            emit_event(
                event_type="document_ocr_failed",
                source="DocumentIntelligence",
                payload={"filename": file_path.name, "error": str(exc)},
            )

    def _build_response(self, record: DocumentRecord) -> dict[str, Any]:
        payload = record.to_dict()
        payload["chunks_added"] = int(record.chunk_count)
        return payload

    async def retry_pending_ocr(self) -> None:
        """
        Background job: find documents in 'ocr_pending' state and re-trigger enrichment.
        """
        pending = [
            r
            for r in self.document_registry.list_documents(limit=50)
            if r.ingest_status == "ocr_pending"
        ]
        if not pending:
            return

        logger.info("Retrying pending OCR for %d documents", len(pending))
        for record in pending:
            # We need to re-analyze to get the list of image pages
            # Since index_document is thread-bound and does a lot of work,
            # we just call ingest_path which handles the deduplication and task creation.
            # However, ingest_path is async.
            file_path = self.settings.data_dir / "LiveFolder" / record.source
            if not file_path.exists():
                logger.warning("Retry target file not found", source=record.source)
                continue

            # Re-running ingest_path is safe because it dedups Chroma/Sparse before re-indexing.
            # For ocr_pending, it will re-run the VLM phase.
            await self.ingest_path(file_path)
