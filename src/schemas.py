from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.config import AetherForgeSettings

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from src.core.orchestrator import ForensicOrchestrator
    from src.guardrails.silicon_colosseum import SiliconColosseum
    from src.learning.history_manager import HistoryManager
    from src.learning.replay_buffer import ReplayBuffer
    from src.meta_agent import MetaAgent


class AppState:
    settings: AetherForgeSettings
    meta_agent: MetaAgent
    replay_buffer: ReplayBuffer
    history_manager: HistoryManager
    colosseum: SiliconColosseum
    orchestrator: ForensicOrchestrator | None = None
    scheduler: AsyncIOScheduler
    vector_store: Any
    sparse_index: Any
    export_engine: Any
    startup_ms: float
    selected_vlm_id: str = "smolvlm-256m"
    streamsync_rss_feeds: list[str] = []
    directory_watcher: Any = None
    system_location: str | None = None
    sync_manager: Any = None


class ChatRequest(BaseModel):
    session_id: str
    module: str
    message: str
    xray_mode: bool = False
    protocol: str = "legacy"
    context: dict[str, Any] = Field(default_factory=dict)


class ChatCitation(BaseModel):
    source: str
    page: int | str | None = None
    section: str | None = None
    snippet: str | None = None
    kind: str = "document"
    label: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    module: str
    latency_ms: float
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    policy_decisions: list[dict[str, Any]] = Field(default_factory=list)
    causal_graph: dict[str, Any] | None = None
    faithfulness_score: float | None = None
    reasoning_trace: str | None = None
    answer_text: str | None = None
    citations: list[ChatCitation] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)


class RefineRequest(BaseModel):
    text: str


class RSSFeedRequest(BaseModel):
    url: str
