from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from src.config import AetherForgeSettings, get_settings
from src.settings_store import load_saved_settings

logger = structlog.get_logger("aetherforge.core.container")


class Container:
    """
    Central dependency injection and lifecycle container for AetherForge.
    Enables better modularity and testability by managing component creation and wiring.
    """

    _instance: Container | None = None

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._services: dict[str, Any] = {}
            self._modules: dict[str, Any] = {}
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
        import uuid

        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        from src.guardrails.silicon_colosseum import SiliconColosseum
        from src.learning.history_manager import HistoryManager
        from src.learning.replay_buffer import ReplayBuffer
        from src.meta_agent import MetaAgent
        from src.modules.document_registry import DocumentRegistry
        from src.modules.export_engine import ExportEngine
        from src.modules.ragforge.sparse_index import SparseIndex
        from src.modules.session_store import SessionStore
        from src.modules.sync.event_log import EventLog
        from src.modules.sync.sync_manager import SyncManager
        from src.services.document_intelligence import DocumentIntelligenceService

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
            persist_directory=str(self.settings.chroma_path), embedding_function=embeddings
        )
        self.register_service("vector_store", vector_store)

        # Sparse Index
        sparse_index = SparseIndex(db_path=self.settings.data_dir / "sparse_index.db")
        self.register_service("sparse_index", sparse_index)

        # Document Registry
        document_registry = DocumentRegistry(db_path=self.settings.data_dir / "document_registry.db")
        self.register_service("document_registry", document_registry)

        # Session Store & Export Engine
        session_store = SessionStore(
            db_path=self.settings.data_dir / "sessions.db",
            key_file=self.settings.sqlcipher_key_file,
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
            replay_buffer=self.get_service("replay_buffer"),
            export_engine=self.get_service("export_engine"),
        )
        await meta_agent.initialize()
        self.register_service("meta_agent", meta_agent)

        # Persisted runtime selections
        saved_settings = load_saved_settings()
        app_state.selected_chat_model = str(
            saved_settings.get("SELECTED_CHAT_MODEL", "bitnet-b1.58-2b")
        )
        app_state.selected_vlm_id = str(saved_settings.get("SELECTED_VLM_ID", "smolvlm-256m"))
        meta_agent.selected_chat_model = app_state.selected_chat_model

        # Document Intelligence
        document_intelligence = DocumentIntelligenceService(
            settings=self.settings,
            vector_store=vector_store,
            sparse_index=sparse_index,
            document_registry=document_registry,
            selected_vlm_id_getter=lambda: app_state.selected_vlm_id,
        )
        self.register_service("document_intelligence", document_intelligence)

        # 7. Sync Manager
        node_id_file = self.settings.data_dir / "node_id.txt"
        if node_id_file.exists():
            node_id = node_id_file.read_text().strip()
        else:
            node_id = str(uuid.uuid4())[:8]
            node_id_file.write_text(node_id)

        sync_event_log = EventLog(db_path=self.settings.data_dir / "sync_events.db")
        sync_manager = SyncManager(
            node_id=node_id, port=self.settings.aetherforge_port, event_log=sync_event_log
        )
        self.register_service("sync_manager", sync_manager)
        app_state.sync_manager = sync_manager
        await sync_manager.start()

        # 8. Module Plugins
        from src.modules.analytics.module import AnalyticsModule
        from src.modules.core_tools import CoreModule
        from src.modules.localbuddy.module import LocalBuddyModule
        from src.modules.ragforge.module import RagForgeModule
        from src.modules.streamsync.module import StreamSyncModule
        from src.modules.sync.module import SyncModule
        from src.modules.tunelab.module import TuneLabModule
        from src.modules.watchtower.module import WatchTowerModule

        self._modules["core"] = CoreModule(self.settings)
        self._modules["watchtower"] = WatchTowerModule()
        self._modules["streamsync"] = StreamSyncModule()
        self._modules["analytics"] = AnalyticsModule()
        self._modules["ragforge"] = RagForgeModule(meta_agent._get_cognitive_rag())
        self._modules["tunelab"] = TuneLabModule(self.settings, self.get_service("replay_buffer"))
        self._modules["localbuddy"] = LocalBuddyModule()
        self._modules["sync"] = SyncModule(sync_manager)

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
        app_state.document_registry = self.get_service("document_registry")
        app_state.document_intelligence = self.get_service("document_intelligence")
        
        # 9. Bootstrap System Knowledge
        await self._bootstrap_system_knowledge()

    async def _bootstrap_system_knowledge(self) -> None:
        """Automatically index the application manual for self-knowledge."""
        manual_path = self.settings.system_manual_path
        if not manual_path.exists():
            manual_path = Path("AetherForge_User_Manual_v1.0.md") # Fallback to root
            
        if manual_path.exists():
            logger.info("Bootstrapping system knowledge from manual", path=str(manual_path))
            doc_intel = self.get_service("document_intelligence")
            try:
                # We use ingest_path which handles mtime-based idempotency
                await doc_intel.ingest_path(manual_path)
            except Exception as e:
                logger.error("Failed to bootstrap system knowledge", error=str(e))
        else:
            logger.warning("System manual not found; AI self-knowledge will be limited", path=str(manual_path))

    async def shutdown_all(self) -> None:
        """Gracefully shut down all services."""
        logger.info("Shutting down services in Container")

        # 1. Sync Manager
        if "sync_manager" in self._services:
            try:
                await self._services["sync_manager"].stop()
                logger.info("SyncManager stopped")
            except Exception as e:
                logger.error("Error stopping SyncManager", error=str(e))

        # 2. Replay Buffer
        if "replay_buffer" in self._services:
            try:
                await self._services["replay_buffer"].flush()
                logger.info("ReplayBuffer flushed")
            except Exception as e:
                logger.error("Error flushing ReplayBuffer", error=str(e))

        # 3. Module Plugins
        for name, mod in self._modules.items():
            try:
                if hasattr(mod, "shutdown"):
                    await mod.shutdown()
                logger.info("Module shut down", module=name)
            except Exception as e:
                logger.error("Error shutting down module", module=name, error=str(e))


container = Container()
