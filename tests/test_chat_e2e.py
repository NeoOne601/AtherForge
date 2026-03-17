import pytest
import sys
import types
from fastapi.testclient import TestClient


def _install_apscheduler_stub() -> None:
    if "apscheduler.schedulers.asyncio" in sys.modules:
        return

    apscheduler = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    triggers = types.ModuleType("apscheduler.triggers")
    cron_mod = types.ModuleType("apscheduler.triggers.cron")

    class DummyScheduler:
        def add_job(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def shutdown(self, wait: bool = False):
            return None

    class DummyCronTrigger:
        def __init__(self, *args, **kwargs):
            pass

    asyncio_mod.AsyncIOScheduler = DummyScheduler
    cron_mod.CronTrigger = DummyCronTrigger
    sys.modules["apscheduler"] = apscheduler
    sys.modules["apscheduler.schedulers"] = schedulers
    sys.modules["apscheduler.schedulers.asyncio"] = asyncio_mod
    sys.modules["apscheduler.triggers"] = triggers
    sys.modules["apscheduler.triggers.cron"] = cron_mod


_install_apscheduler_stub()

from src.app_factory import create_app

@pytest.fixture
def client():
    """Provides a TestClient with triggered lifespan."""
    with TestClient(create_app()) as c:
        yield c

def test_explicit_session_creation_and_empty_restore(client):
    response = client.post("/api/v1/sessions", json={"module": "localbuddy"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"].startswith("localbuddy:")

    messages = client.get(f"/api/v1/sessions/{data['id']}/messages")
    assert messages.status_code == 200
    assert messages.json() == []

def test_chat_e2e_basic(client):
    """Verify the canonical chat path returns a valid response."""
    payload = {
        "message": "Hello, computer!",
        "module": "localbuddy",
        "session_id": "test-session-123",
        "xray_mode": False
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["session_id"] == "localbuddy:test-session-123"
    assert data["module"] == "localbuddy"
    assert "reasoning_summary" in data
    assert isinstance(data["suggestions"], list)

def test_chat_e2e_xray(client):
    """Verify xray_mode returns a causal graph."""
    payload = {
        "message": "Analyze system performance",
        "module": "watchtower",
        "session_id": "test-session-xray",
        "xray_mode": True
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "causal_graph" in data
    assert data["causal_graph"] is not None

def test_chat_ws_streaming(client):
    """Verify WebSocket streaming yields structured events and persists the turn."""
    session = client.post("/api/v1/sessions", json={"module": "localbuddy"}).json()
    session_id = session["id"]

    with client.websocket_connect(f"/api/v1/ws/{session_id}") as websocket:
        websocket.send_json({
            "message": "Stream a thought and a response",
            "session_id": session_id,
            "module": "localbuddy",
            "xray_mode": True
        })

        events = []
        for _ in range(100):
            try:
                data = websocket.receive_json()
                events.append(data)
                if data["type"] == "done":
                    break
            except Exception:
                break
        
        assert len(events) > 0
        assert any(e["type"] in ["token", "reasoning", "meta"] for e in events)
        done = next(e for e in events if e["type"] == "done")
        assert done["session_id"] == session_id
        assert "response" in done
        assert "reasoning_summary" in done
        assert isinstance(done["suggestions"], list)

    restored = client.get(f"/api/v1/sessions/{session_id}/messages")
    assert restored.status_code == 200
    messages = restored.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "reasoning_summary" in messages[1]["metadata"]
