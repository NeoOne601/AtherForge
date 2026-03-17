import sys
import os
import asyncio
from typing import AsyncGenerator, Any
from unittest.mock import MagicMock

# --- Mocking Dependencies ---
mock_langchain = MagicMock()
sys.modules["langchain_core"] = mock_langchain
sys.modules["langchain_core.messages"] = mock_langchain
sys.modules["pydantic"] = MagicMock()
sys.modules["structlog"] = MagicMock()
sys.modules["src.config"] = MagicMock()
sys.modules["src.guardrails.silicon_colosseum"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath("src"))

# Now we can import
from src.chat_contract import sanitize_output, split_reasoning_trace
from src.meta_agent import StreamSanitizer

async def test_sanitization():
    print("--- Testing Sanitization ---")
    
    # Test 1: Aggressive horizontal whitespace collapse
    input_text = "Hello    world    this  is a   test."
    expected = "Hello world this is a test."
    assert sanitize_output(input_text) == expected, f"Expected '{expected}', got '{sanitize_output(input_text)}'"
    print("Test 1 Passed: Horizontal whitespace collapsed.")

    # Test 2: Newline preservation
    input_text = "Line 1\nLine 2\n  Line 3  "
    expected = "Line 1\nLine 2\nLine 3"
    assert sanitize_output(input_text) == expected, f"Expected '{expected}', got '{sanitize_output(input_text)}'"
    print("Test 2 Passed: Newlines preserved.")

    # Test 3: Stream mode (no stripping)
    chunk = "  hello  "
    assert sanitize_output(chunk, is_stream=True) == chunk
    print("Test 3 Passed: Stream mode preserves spaces.")

async def test_reasoning():
    print("\n--- Testing Reasoning Extraction ---")
    
    # Test 1: Standard tags
    text = "<think>I am thinking.</think>Hello user!"
    reasoning, answer = split_reasoning_trace(text)
    assert reasoning == "I am thinking."
    assert answer == "Hello user!"
    print("Test 1 Passed: Standard tags handled.")

    # Test 2: Partial closing tag
    text = "<think>Still thinking...</think"
    reasoning, answer = split_reasoning_trace(text)
    assert reasoning == "Still thinking..."
    # Note: answer will be sanitized_output of whatever was before <think> or empty
    print("Test 2 Passed: Partial closing tag handled.")

    # Test 3: No closing tag (streaming-like)
    text = "<think>I just started thinking"
    reasoning, answer = split_reasoning_trace(text)
    assert reasoning == "I just started thinking"
    print("Test 3 Passed: Unclosed think block handled.")

async def test_stream_sanitizer():
    print("\n--- Testing StreamSanitizer ---")
    
    async def mock_generator():
        chunks = ["<think>", "Thinking ", "hard...", "</think>", "Final answer."]
        for c in chunks:
            yield c
            await asyncio.sleep(0.01)

    sanitizer = StreamSanitizer(mock_generator())
    results = []
    async for event in sanitizer:
        results.append(event)
    
    # Concatenate results
    reasoning = "".join([e["content"] for e in results if e["type"] == "reasoning"])
    tokens = "".join([e["content"] for e in results if e["type"] == "token"])
    
    print(f"Reasoning extracted: '{reasoning}'")
    print(f"Tokens extracted: '{tokens}'")
    
    assert "Thinking hard" in reasoning
    assert "Final answer" in tokens
    print("Test StreamSanitizer Passed.")

if __name__ == "__main__":
    try:
        asyncio.run(test_sanitization())
        asyncio.run(test_reasoning())
        asyncio.run(test_stream_sanitizer())
        print("\nALL RECOVERY TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
