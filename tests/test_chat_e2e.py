import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.schemas import ChatResponse
import json

@pytest.fixture
def client():
    """Provides a TestClient with triggered lifespan."""
    with TestClient(app) as c:
        yield c

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
    """Verify WebSocket streaming yields structured events."""
    with client.websocket_connect("/api/v1/ws/test-session-stream") as websocket:
        websocket.send_json({
            "message": "Stream a thought and a response",
            "module": "localbuddy",
            "xray_mode": True
        })
        
        # We expect at least one thought or token and a done event
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
        # Check for type token or thought or meta
        assert any(e["type"] in ["token", "thought", "meta"] for e in events)
        assert any(e["type"] == "done" for e in events)
