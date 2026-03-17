import os
import sys

# Ensure project root is in path for IDE/shell resolution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chat_contract import sanitize_output, split_reasoning_trace

def test_sanitization():
    print("--- Testing Global Sanitizer Guard ---")
    
    # Test 1: Thinking tags
    input1 = "<think>reasoning</think>Hello world"
    reasoning1, answer1 = split_reasoning_trace(input1)
    assert answer1 == "Hello world"
    assert reasoning1 == "reasoning"
    print("Test 1: Simple tags - PASSED")
    
    # Test 2: Stray tags
    input2 = "</think>Greetings <think> more"
    answer2 = sanitize_output(input2)
    assert answer2 == "Greetings more"
    print("Test 2: Stray tags - PASSED")
    
    # Test 3: Tool JSON leak
    input3 = 'Here is the results: {"name": "search_web", "arguments": {"query": "test"}} Done.'
    answer3 = sanitize_output(input3)
    print(f"DEBUG: answer3=[[{answer3}]]")
    assert answer3 == "Here is the results: Done."
    print("Test 3: Tool JSON leak - PASSED")
    
    # Test 4: Nested tags and JSON
    input4 = "<think>{\"name\": \"internal\"}</think>Actual answer"
    reasoning4, answer4 = split_reasoning_trace(input4)
    assert answer4 == "Actual answer"
    assert reasoning4 == '{"name": "internal"}'
    print("Test 4: Nested JSON in reasoning - PASSED")
    
    # Test 5: Complex markdown JSON
    input5 = "Answer: ```json\n{\"name\": \"search_web\", \"arguments\": {}}\n```"
    answer5 = sanitize_output(input5)
    assert answer5 == "Answer:"
    print("Test 5: Markdown JSON - PASSED")

    print("\n--- ALL GLOBAL GUARD TESTS PASSED ---")

if __name__ == "__main__":
    test_sanitization()
