import asyncio
import time
from pathlib import Path
from typing import Any

import structlog
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.modules.ragforge_indexer import index_document
from src.modules.streamsync.graph import emit_event

logger = structlog.get_logger("aetherforge.streamsync.watcher")


class AutoIndexHandler(FileSystemEventHandler):
    def __init__(
        self, watch_dir: Path, loop: asyncio.AbstractEventLoop, vector_store: Any, sparse_index: Any
    ):
        self.watch_dir = watch_dir
        self.loop = loop
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self._lock = asyncio.Lock()  # Prevent concurrent heavy indexing tasks

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
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
                if self.vector_store is None or self.sparse_index is None:
                    logger.error("Vector components not provided to watcher. Delaying index.")
                    return

                result = await asyncio.to_thread(
                    index_document, file_path, self.vector_store, self.sparse_index
                )

                chunks_added = (
                    result.get("chunks_added", 0) if isinstance(result, dict) else int(result)
                )

                logger.info(
                    "StreamSync Auto-Indexed '%s' — %d chunks", file_path.name, chunks_added
                )

                # Emit event to the StreamSync HUD
                emit_event(
                    event_type="document_indexed",
                    source="DirectoryWatcher",
                    payload={
                        "filename": file_path.name,
                        "chunks": chunks_added,
                        "status": "success",
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
        self, watch_dir: Path, loop: asyncio.AbstractEventLoop, vector_store: Any, sparse_index: Any
    ):
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.loop = loop
        self.observer = Observer()
        self.handler = AutoIndexHandler(self.watch_dir, self.loop, vector_store, sparse_index)

    def start(self):
        # 1. Start a throttled boot-sweep task
        asyncio.run_coroutine_threadsafe(self._boot_sweep(), self.loop)

        # 2. Watch for any new incoming files
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        logger.info("StreamSync Directory Watcher started on %s", self.watch_dir)

    async def _boot_sweep(self):
        """Throttled ingestion of existing files to prevent CPU/RAM spikes on startup."""
        print(f"DEBUG: _boot_sweep: checking {self.watch_dir}", flush=True)
        existing_files = [
            f for f in self.watch_dir.iterdir() if f.is_file() and not f.name.startswith(".")
        ]
        print(
            f"DEBUG: _boot_sweep: found {len(existing_files)} files: {[f.name for f in existing_files]}",
            flush=True,
        )
        if not existing_files:
            return

        logger.info("StreamSync Boot-Sweep: scheduling %d files (5s interval)", len(existing_files))
        for i, filepath in enumerate(existing_files):
            # We don't need run_coroutine_threadsafe here because _boot_sweep
            # is ALREADY running on the main event loop.
            print(f"DEBUG: _boot_sweep: indexing {filepath.name}", flush=True)
            await self.handler.async_index(filepath)

            # Wait 5 seconds between each background document to keep system responsive
            if i < len(existing_files) - 1:
                await asyncio.sleep(5.0)

    def stop(self):
        self.observer.stop()
        self.observer.join()
        logger.info("StreamSync Directory Watcher stopped.")
