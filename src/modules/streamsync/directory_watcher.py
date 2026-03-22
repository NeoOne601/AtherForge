import asyncio
import time
from pathlib import Path
from typing import Any

import structlog # type: ignore
from watchdog.events import FileSystemEventHandler # type: ignore
from watchdog.observers import Observer # type: ignore

from src.modules.streamsync.graph import emit_event # type: ignore

logger = structlog.get_logger("aetherforge.streamsync.watcher")


class AutoIndexHandler(FileSystemEventHandler):
    def __init__(
        self,
        watch_dir: Path,
        loop: asyncio.AbstractEventLoop,
        app_state: object,
    ):
        self.watch_dir = watch_dir
        self.loop = loop
        self.app_state = app_state
        self._lock = asyncio.Lock()  # Prevent concurrent heavy indexing tasks

    def on_created(self, event):
        if event.is_directory:
            return
        if Path(event.src_path).name.startswith('.'):
            return
        self._handle_event(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if Path(event.dest_path).name.startswith('.'):
            return
        # src_path is the old path, dest_path is the new path inside our watch_dir
        self._handle_event(event.dest_path)

    def _handle_event(self, path_str: str):
        file_path = Path(path_str)

        # Give the filesystem a moment to finish writing the file to disk
        # before we try to open and read it for indexing.
        time.sleep(1.0)

        logger.info("StreamSync Directory Watcher detected change: %s", file_path.name)

        # We use the explicitly passed main event loop instead of get_running_loop()
        # because this handler executes on a background watchdog thread.
        asyncio.run_coroutine_threadsafe(self.async_index(file_path), self.loop)

    async def async_index(self, file_path: Path) -> None:
        async with self._lock:
            try:
                document_intelligence = getattr(self.app_state, "document_intelligence", None)
                if document_intelligence is None:
                    logger.error("Document intelligence service not ready. Delaying index.")
                    return

                result = await document_intelligence.ingest_path(file_path)
                chunks_added = int(result.get("chunks_added", 0))
                status = str(result.get("ingest_status", "ready"))

                logger.info(
                    "StreamSync Auto-Indexed '%s' — %d chunks (%s)",
                    file_path.name,
                    chunks_added,
                    status,
                )

                # Emit event to the StreamSync HUD
                emit_event(
                    event_type="document_indexed",
                    source="DirectoryWatcher",
                    payload={
                        "filename": file_path.name,
                        "chunks": chunks_added,
                        "status": status,
                    },
                )
            except Exception as e:
                logger.error("Failed to auto-index dropped file %s: %s", file_path.name, e)
                emit_event(
                    event_type="document_index_failed",
                    source="DirectoryWatcher",
                    payload={"filename": file_path.name, "error": str(e)},
                )


class StreamSyncDirectoryWatcher:
    """Manages the watchdog observer for the AetherForge-Live folder."""

    def __init__(
        self,
        watch_dir: Path,
        loop: asyncio.AbstractEventLoop,
        app_state: object,
    ):
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.loop = loop
        self.observer = Observer()
        from concurrent.futures import Future
        self.handler = AutoIndexHandler(self.watch_dir, self.loop, app_state)
        self._boot_task: Future[Any] | None = None
        self._is_stopping = False

    def start(self):
        # 1. Start a throttled boot-sweep task
        self._boot_task = asyncio.run_coroutine_threadsafe(self._boot_sweep(), self.loop)

        # 2. Watch for any new incoming files
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        logger.info("StreamSync Directory Watcher started on %s", self.watch_dir)

    async def _boot_sweep(self):
        """Throttled ingestion of existing files to prevent CPU/RAM spikes on startup."""
        if self._is_stopping:
            return

        existing_files = [
            f for f in self.watch_dir.iterdir() if f.is_file() and not f.name.startswith(".")
        ]
        if not existing_files:
            return

        logger.info("StreamSync Boot-Sweep: scheduling %d files (5s interval)", len(existing_files))
        for i, filepath in enumerate(existing_files):
            if self._is_stopping:
                break

            await self.handler.async_index(filepath)

            # Wait 5 seconds between each background document to keep system responsive
            if i < len(existing_files) - 1:
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    break

    def stop(self):
        self._is_stopping = True
        # Cancel the boot sweep future if it hasn't finished
        boot_task = self._boot_task
        if boot_task is not None and not boot_task.done():
            boot_task.cancel()
            
        self.observer.stop()
        try:
            # Short timeout to prevent hanging the whole shutdown process
            self.observer.join(timeout=2.0)
        except Exception:
            pass
        logger.info("StreamSync Directory Watcher stopped.")
