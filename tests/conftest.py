import pytest
import os
import sys
import types
from unittest.mock import MagicMock
import numpy as np


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _install_global_test_stubs() -> None:
    watchdog_mod = _ensure_module("watchdog")
    watchdog_events = _ensure_module("watchdog.events")
    watchdog_observers = _ensure_module("watchdog.observers")

    class FileSystemEventHandler:
        pass

    class Observer:
        def schedule(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def join(self, timeout=None):
            return None

    watchdog_events.FileSystemEventHandler = FileSystemEventHandler
    watchdog_observers.Observer = Observer
    watchdog_mod.events = watchdog_events
    watchdog_mod.observers = watchdog_observers

    bs4_mod = _ensure_module("bs4")

    class BeautifulSoup:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def find_all(self, *args, **kwargs):
            return []

    bs4_mod.BeautifulSoup = BeautifulSoup

    websockets_mod = _ensure_module("websockets")

    class _DummyWebSocketConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, *args, **kwargs):
            return None

        async def recv(self):
            return "{}"

    def connect(*args, **kwargs):
        return _DummyWebSocketConnection()

    websockets_mod.connect = connect

    zeroconf_mod = _ensure_module("zeroconf")

    class Zeroconf:
        def __init__(self, *args, **kwargs):
            pass

        def register_service(self, *args, **kwargs):
            return None

        def unregister_service(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def get_service_info(self, *args, **kwargs):
            return None

    class ServiceInfo:
        def __init__(self, *args, **kwargs):
            self.addresses = kwargs.get("addresses", [])
            self.port = kwargs.get("port", 0)

    class ServiceBrowser:
        def __init__(self, *args, **kwargs):
            pass

    class _IPVersion:
        V4Only = "V4Only"

    class _ServiceStateChange:
        Added = "Added"
        Removed = "Removed"

    class NonUniqueNameException(Exception):
        pass

    zeroconf_mod.Zeroconf = Zeroconf
    zeroconf_mod.ServiceInfo = ServiceInfo
    zeroconf_mod.ServiceBrowser = ServiceBrowser
    zeroconf_mod.IPVersion = _IPVersion
    zeroconf_mod.ServiceStateChange = _ServiceStateChange
    zeroconf_mod.NonUniqueNameException = NonUniqueNameException

    multipart_mod = _ensure_module("multipart")
    multipart_mod.__version__ = "0.0-test"
    multipart_submod = _ensure_module("multipart.multipart")

    def parse_options_header(value):
        return value, {}

    multipart_submod.parse_options_header = parse_options_header
    multipart_mod.multipart = multipart_submod

    langchain_core = _ensure_module("langchain_core")
    messages_mod = _ensure_module("langchain_core.messages")
    documents_mod = _ensure_module("langchain_core.documents")

    class _BaseMessage:
        def __init__(self, content=None, **kwargs):
            self.content = content
            self.kwargs = kwargs

    class SystemMessage(_BaseMessage):
        pass

    class HumanMessage(_BaseMessage):
        pass

    class AIMessage(_BaseMessage):
        pass

    class Document:
        def __init__(self, page_content="", metadata=None, **kwargs):
            self.page_content = page_content
            self.metadata = metadata or {}
            self.kwargs = kwargs

    messages_mod.SystemMessage = SystemMessage
    messages_mod.HumanMessage = HumanMessage
    messages_mod.AIMessage = AIMessage
    documents_mod.Document = Document
    langchain_core.messages = messages_mod
    langchain_core.documents = documents_mod


_install_global_test_stubs()


@pytest.fixture(autouse=True)
def mock_heavy_deps(monkeypatch, tmp_path):
    """
    Globally mock heavy dependencies (Torch, LLMs, Embeddings) for all tests.
    Also ensures each test uses a clean temporary data directory.
    """
    # 1. Setup clean data directory
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(test_data_dir))
    monkeypatch.setenv("BITNET_MODEL_PATH", str(test_data_dir / "mock_model.gguf"))
    monkeypatch.setenv("AETHERFORGE_ENV", "test")
    try:
        from src.config import get_settings

        get_settings.cache_clear()
    except Exception:
        pass

    # 2. Mock HuggingFaceEmbeddings
    class MockEmbeddings:
        def embed_documents(self, texts):
            return [np.zeros(384).tolist() for _ in texts]
        def embed_query(self, text):
            return np.zeros(384).tolist()

    mock_hf = MagicMock(return_value=MockEmbeddings())
    _ensure_module("langchain_huggingface")
    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", mock_hf, raising=False)

    # 3. Mock Llama (llama-cpp-python)
    mock_llama = MagicMock()
    _ensure_module("llama_cpp")
    monkeypatch.setattr("llama_cpp.Llama", mock_llama, raising=False)
    
    # 4. Mock Chroma (to avoid disk I/O and sqlite issues in tests)
    mock_chroma = MagicMock()
    _ensure_module("langchain_chroma")
    monkeypatch.setattr("langchain_chroma.Chroma", mock_chroma, raising=False)
    
    yield

    try:
        from src.config import get_settings

        get_settings.cache_clear()
    except Exception:
        pass
