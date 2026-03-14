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
