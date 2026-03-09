from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional, List, TYPE_CHECKING
from src.config import AetherForgeSettings

if TYPE_CHECKING:
    from src.meta_agent import MetaAgent
    from src.learning.replay_buffer import ReplayBuffer
    from src.learning.history_manager import HistoryManager
    from src.guardrails.silicon_colosseum import SiliconColosseum
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

class AppState:
    settings: AetherForgeSettings
    meta_agent: MetaAgent
    replay_buffer: ReplayBuffer
    history_manager: HistoryManager
    colosseum: SiliconColosseum
    scheduler: AsyncIOScheduler
    vector_store: Any
    sparse_index: Any
    export_engine: Any
    startup_ms: float
    selected_vlm_id: str = "smolvlm-256m"
    streamsync_rss_feeds: List[str] = []
    directory_watcher: Any = None
    system_location: Optional[str] = None
    sync_manager: Any = None

class ChatRequest(BaseModel):
    session_id: str
    module: str
    message: str
    xray_mode: bool = False
    context: dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    session_id: str
    response: str
    module: str
    latency_ms: float
    tool_calls: List[dict[str, Any]] = []
    policy_decisions: List[dict[str, Any]] = []
    causal_graph: Optional[dict[str, Any]] = None
    faithfulness_score: Optional[float] = None

class RefineRequest(BaseModel):
    text: str

class RSSFeedRequest(BaseModel):
    url: str
