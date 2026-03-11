from __future__ import annotations
import asyncio
import structlog
from typing import Any, Dict, Optional, List

from src.config import get_settings, AetherForgeSettings

logger = structlog.get_logger("aetherforge.core.container")

class Container:
    """
    Central dependency injection and lifecycle container for AetherForge.
    Enables better modularity and testability by managing component creation and wiring.
    """
    _instance: Optional[Container] = None

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._services: Dict[str, Any] = {}
            self._modules: Dict[str, Any] = {}
            self.settings: AetherForgeSettings = get_settings()
            self._initialized = True

    def register_service(self, name: str, service: Any) -> None:
        """Add a service to the container."""
        self._services[name] = service
        logger.debug("Service registered", name=name)

    def get_service(self, name: str) -> Any:
        """Retrieve a service from the container."""
        if name not in self._services:
            logger.error("Service not found", name=name)
            raise ValueError(f"Service '{name}' not found.")
        return self._services[name]

    async def initialize_all(self, app_state: Any) -> None:
        """Initialize all core services and wire them together."""
        from src.learning.replay_buffer import ReplayBuffer
        from src.learning.history_manager import HistoryManager
        from src.guardrails.silicon_colosseum import SiliconColosseum
        from src.modules.session_store import SessionStore
        from src.modules.export_engine import ExportEngine
        from src.meta_agent import MetaAgent
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.modules.ragforge.sparse_index import SparseIndex

        logger.info("Initializing services in Container")

        # Replay Buffer & History
        self.register_service("replay_buffer", ReplayBuffer(self.settings))
        await self.get_service("replay_buffer").initialize()
        self.register_service("history_manager", HistoryManager(self.settings))

        # Embeddings & Vector Store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.register_service("embeddings", embeddings)
        vector_store = Chroma(
            persist_directory=str(self.settings.chroma_path), 
            embedding_function=embeddings
        )
        self.register_service("vector_store", vector_store)

        # Sparse Index
        sparse_index = SparseIndex(db_path=self.settings.data_dir / "sparse_index.db")
        self.register_service("sparse_index", sparse_index)

        # Session Store & Export Engine
        session_store = SessionStore(
            db_path=self.settings.data_dir / "sessions.db", 
            key_file=self.settings.sqlcipher_key_file
        )
        self.register_service("session_store", session_store)
        self.register_service("export_engine", ExportEngine(session_store))

        # Guardrails
        colosseum = SiliconColosseum(self.settings)
        await colosseum.initialize()
        self.register_service("colosseum", colosseum)

        # Meta Agent
        meta_agent = MetaAgent(
            self.settings, 
            colosseum, 
            vector_store=vector_store, 
            sparse_index=self.get_service("sparse_index"),
            replay_buffer=self.get_service("replay_buffer")
        )
        await meta_agent.initialize()
        self.register_service("meta_agent", meta_agent)

        # 8. Module Plugins
        from src.modules.core_tools import CoreModule
        from src.modules.watchtower.module import WatchTowerModule
        from src.modules.streamsync.module import StreamSyncModule
        from src.modules.analytics.module import AnalyticsModule
        
        self._modules["core"] = CoreModule(self.settings)
        self._modules["watchtower"] = WatchTowerModule()
        self._modules["streamsync"] = StreamSyncModule()
        self._modules["analytics"] = AnalyticsModule()
        
        for mod in self._modules.values():
            await mod.initialize()
            mod.register_tools()
            logger.info("Module Plugin initialized and tools registered", module=mod.name)

        # Sync AppState with Container (backwards compatibility)
        app_state.settings = self.settings
        app_state.replay_buffer = self.get_service("replay_buffer")
        app_state.history_manager = self.get_service("history_manager")
        app_state.vector_store = self.get_service("vector_store")
        app_state.sparse_index = self.get_service("sparse_index")
        app_state.session_store = self.get_service("session_store")
        app_state.export_engine = self.get_service("export_engine")
        app_state.colosseum = self.get_service("colosseum")
        app_state.meta_agent = self.get_service("meta_agent")

    async def shutdown_all(self) -> None:
        """Gracefully shut down all services."""
        logger.info("Shutting down services in Container")
        if "replay_buffer" in self._services:
            await self._services["replay_buffer"].flush()

container = Container()
