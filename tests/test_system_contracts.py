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


def test_streamsync_contract_endpoints():
    with TestClient(create_app()) as client:
        state = client.app.state.app_state
        state.streamsync_rss_feeds = ["https://example.com/rss.xml"]

        rss = client.get("/api/v1/streamsync/rss")
        assert rss.status_code == 200
        assert rss.json()["feeds"] == ["https://example.com/rss.xml"]

        events = client.get("/api/v1/events/stream?limit=10")
        assert events.status_code == 200
        assert isinstance(events.json(), list)


def test_ragforge_documents_use_registry_and_selection_round_trip():
    with TestClient(create_app()) as client:
        registry = client.app.state.app_state.document_registry
        record = registry.upsert_document(
            source="demo.pdf",
            file_type="pdf",
            ingest_status="ocr_pending",
            parser="pending_vlm",
            chunk_count=0,
            image_pages_pending=12,
            last_error=None,
            selected=True,
        )

        listed = client.get("/api/v1/ragforge/documents")
        assert listed.status_code == 200
        documents = listed.json()["documents"]
        payload = next(doc for doc in documents if doc["document_id"] == record.document_id)
        assert payload["name"] == "demo.pdf"
        assert payload["status"] == "ocr_pending"
        assert payload["image_pages_pending"] == 12

        updated = client.patch(
            f"/api/v1/ragforge/documents/{record.document_id}",
            json={"selected": False},
        )
        assert updated.status_code == 200
        assert updated.json()["selected"] is False
