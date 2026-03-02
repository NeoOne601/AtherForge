# AetherForge v1.0 — tests/test_replay_buffer.py
# Unit tests for the encrypted replay buffer.
import asyncio
import pytest
from pathlib import Path

from src.config import AetherForgeSettings
from src.learning.replay_buffer import ReplayBuffer


@pytest.fixture
def settings(tmp_path):
    s = AetherForgeSettings(
        aetherforge_env="test",
        data_dir=tmp_path,
        replay_buffer_path=tmp_path / "test_replay.parquet",
        sqlcipher_key_file=tmp_path / ".test_key",
    )
    s.ensure_data_dirs()
    return s


@pytest.fixture
async def buffer(settings):
    buf = ReplayBuffer(settings)
    await buf.initialize()
    return buf


@pytest.mark.asyncio
async def test_initialize_creates_file(settings):
    buf = ReplayBuffer(settings)
    await buf.initialize()
    assert settings.replay_buffer_path.exists()


@pytest.mark.asyncio
async def test_record_and_flush(settings):
    buf = ReplayBuffer(settings)
    await buf.initialize()
    buf._flush_threshold = 1  # Flush after every record in test

    rid = await buf.record(
        session_id="s1",
        module="localbuddy",
        prompt="What is OPLoRA?",
        response="OPLoRA prevents catastrophic forgetting via SVD projections.",
        faithfulness_score=0.95,
    )
    assert isinstance(rid, str) and len(rid) == 36  # UUID format
    await buf.flush()
    stats = await buf.get_stats()
    assert stats["total_records"] >= 1


@pytest.mark.asyncio
async def test_sample_returns_records(settings):
    buf = ReplayBuffer(settings)
    await buf.initialize()
    buf._flush_threshold = 1

    for i in range(5):
        await buf.record(
            session_id=f"s{i}",
            module="ragforge",
            prompt=f"Query {i}",
            response=f"Answer {i}",
            faithfulness_score=0.95,
        )
    await buf.flush()

    samples = await buf.sample(n=3, module="ragforge")
    assert len(samples) == 3


@pytest.mark.asyncio
async def test_encryption_key_created(settings):
    buf = ReplayBuffer(settings)
    await buf.initialize()
    assert settings.sqlcipher_key_file.exists()
    assert settings.sqlcipher_key_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_mark_as_used(settings):
    buf = ReplayBuffer(settings)
    await buf.initialize()
    buf._flush_threshold = 1

    rid = await buf.record(
        session_id="s1", module="localbuddy",
        prompt="test", response="test", faithfulness_score=0.95,
    )
    await buf.flush()
    updated = await buf.mark_as_used([rid])
    assert updated == 1

    # Exclude used should return 0
    fresh = await buf.sample(n=10, exclude_used=True)
    assert all(not r.get("is_used_for_training") for r in fresh)


@pytest.mark.asyncio
async def test_get_stats_empty_buffer(settings):
    buf = ReplayBuffer(settings)
    await buf.initialize()
    stats = await buf.get_stats()
    assert stats["total_records"] == 0
    assert stats["size_mb"] >= 0
