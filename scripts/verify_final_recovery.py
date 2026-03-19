import sys
import os
import asyncio
import re

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.meta_agent import StreamSanitizer
from src.modules.ragforge_indexer import MEMORY_CEILING_PCT
from src.routers.ragforge import router as ragforge_router

async def test_stream_sanitizer_thinking_json():
    print("Testing StreamSanitizer thinking-with-json logic...")
    
    async def mock_llm_gen():
        yield "<think>\n"
        yield "I should call a tool.\n"
        yield '{"name": "search_web", "arguments": {"query": "test"}}\n'
        yield "Thinking complete.\n"
        yield "</think>\n"
        yield "Final answer."

    sanitizer = StreamSanitizer(mock_llm_gen())
    results = []
    async for event in sanitizer:
        results.append(event)
        print(f"  Event: {event}")

    # Verify that the JSON was NOT shielded because it was inside <think>
    reasoning_events = [e for e in results if e['type'] == 'reasoning']
    reasoning_content = "".join([e['content'] for e in reasoning_events])
    
    if '{"name": "search_web"' in reasoning_content:
        print("✅ SUCCESS: JSON inside <think> was preserved as reasoning.")
    else:
        print("❌ FAILURE: JSON inside <think> was swallowed or shielded.")

async def test_memory_ceiling():
    print(f"Testing Memory Ceiling: Current value is {MEMORY_CEILING_PCT}%")
    if MEMORY_CEILING_PCT == 92.0:
        print("✅ SUCCESS: Memory ceiling is correctly set to 92.0%")
    else:
        print(f"❌ FAILURE: Memory ceiling is {MEMORY_CEILING_PCT}%, expected 92.0%")

async def test_ragforge_retry_endpoint():
    print("Testing RAGForge router for retry endpoint...")
    routes = [r.path for r in ragforge_router.routes]
    if any("/documents/{document_id}/retry" in r for r in routes):
        print("✅ SUCCESS: Retry endpoint found in RAGForge router.")
    else:
        print("❌ FAILURE: Retry endpoint NOT found.")

if __name__ == "__main__":
    asyncio.run(test_stream_sanitizer_thinking_json())
    asyncio.run(test_memory_ceiling())
    asyncio.run(test_ragforge_retry_endpoint())
