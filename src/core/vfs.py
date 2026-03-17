# AetherForge v1.0 — src/core/vfs.py
# ─────────────────────────────────────────────────────────────────
# Virtual File System (VFS) — Stateful Scratchpad for Deep Research.
# Enables agents to store intermediate findings, offload context,
# and synchronize findings with TuneLab for intelligent learning.
# ─────────────────────────────────────────────────────────────────

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog # type: ignore

logger = structlog.get_logger("aetherforge.vfs")

class VirtualFileSystem:
    """
    A persistent scratchpad per session that stores intermediate research notes.
    Prevents conversation history bloat by offloading findings to a separate state.
    """
    
    def __init__(self, session_id: str, storage_dir: Optional[str] = None):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir) if storage_dir else Path("/tmp/aetherforge/vfs")
        self.notes_file = self.storage_dir / f"{session_id}_vfs.json"
        self._ensure_storage()
        self.notes: List[Dict[str, Any]] = self._load()

    def _ensure_storage(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[Dict[str, Any]]:
        if self.notes_file.exists():
            try:
                with open(self.notes_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to load VFS notes", error=str(e))
        return []

    def _save(self):
        try:
            with open(self.notes_file, "w") as f:
                json.dump(self.notes, f, indent=2)
        except Exception as e:
            logger.error("Failed to save VFS notes", error=str(e))

    def write_note(self, title: str, content: str, source: str = "agent"):
        """Write an intermediate finding or observation to the scratchpad."""
        note = {
            "title": title,
            "content": content,
            "source": source,
            "timestamp": time.time(),
            "id": f"note_{len(self.notes) + 1}"
        }
        self.notes.append(note)
        self._save()
        logger.info("VFS note written", title=title, id=note["id"])

    def get_summary(self) -> str:
        """Provide a concise summary of all findings for the LLM prompt."""
        if not self.notes:
            return "No previous findings in scratchpad."
        
        summary_parts = ["--- VFS SCRATCHPAD SUMMARY ---"]
        for note in self.notes:
            ts = time.strftime("%H:%M:%S", time.localtime(note["timestamp"]))
            summary_parts.append(f"[{ts}] {note['title']}: {note['content'][:200]}...")
        return "\n".join(summary_parts)

    def export_to_tunelab(self) -> Dict[str, Any]:
        """Export current findings for TuneLab self-optimization indexing."""
        logger.info("Exporting VFS state to TuneLab", session_id=self.session_id)
        return {
            "session_id": self.session_id,
            "knowledge_base": self.notes,
            "export_ts": time.time()
        }

    def purge(self):
        """Wipe the scratchpad and delete the file to free up memory/disk."""
        self.notes = []
        if self.notes_file.exists():
            self.notes_file.unlink()
        logger.info("VFS purged", session_id=self.session_id)

    def list_notes(self) -> List[Dict[str, Any]]:
        """Return all notes for UI display or inspection."""
        return self.notes
