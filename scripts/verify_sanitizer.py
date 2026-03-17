
import asyncio
import re
from typing import AsyncGenerator, Any, Optional

# Mocking parts of meta_agent for testing
class AIMessage:
    def __init__(self, content): self.content = content
class SystemMessage:
    def __init__(self, content): self.content = content

class StreamSanitizer:
    def __init__(self, generator: AsyncGenerator[str, None]):
        self.generator = generator
        self.is_thinking = False
        self.shield_active = False 
        self.json_buffer = ""
        self.content_buffer = ""
        self.recovered_tool: Optional[dict[str, Any]] = None
        
    async def __aiter__(self) -> AsyncGenerator[dict[str, Any], None]:
        async for chunk in self.generator:
            self.content_buffer += chunk
            while True:
                if not self.is_thinking:
                    match = re.search(r"<think>", self.content_buffer)
                    if match:
                        pre = self.content_buffer[:match.start()]  # type: ignore
                        if pre: yield {"type": "token", "content": pre}
                        self.is_thinking = True
                        self.content_buffer = self.content_buffer[match.end():]  # type: ignore
                        continue
                    else:
                        # Universal Shielding: matches tools, Actions, and Thinking markers
                        json_match = re.search(r"(?:```json|\{?\s*\"?(?:name|tool_name|search_web|get_weather|get_joke|rag_search)\"?|Action:|Call:|Tool:|Thought:|Assistant:)", self.content_buffer)
                        if json_match:
                             pre = self.content_buffer[:json_match.start()]  # type: ignore
                             if pre: yield {"type": "token", "content": pre}
                             
                             self.shield_active = True
                             self.json_buffer += self.content_buffer[json_match.start():]  # type: ignore
                             self.content_buffer = ""
                             break
                        else:
                             if len(self.content_buffer) > 10:
                                 yield {"type": "token", "content": self.content_buffer[:-10]}  # type: ignore
                                 self.content_buffer = self.content_buffer[-10:]  # type: ignore
                             break
                else:
                    match = re.search(r"</think>", self.content_buffer)
                    if match:
                        trace = self.content_buffer[:match.start()]  # type: ignore
                        if trace: yield {"type": "reasoning", "content": trace}
                        self.is_thinking = False
                        self.content_buffer = self.content_buffer[match.end():]  # type: ignore
                        continue
                    else:
                        if len(self.content_buffer) > 10:
                            yield {"type": "reasoning", "content": self.content_buffer[:-10]}  # type: ignore
                            self.content_buffer = self.content_buffer[-10:]  # type: ignore
                        break
            
            if self.shield_active:
                if len(self.json_buffer) > 1500:
                    self.shield_active = False
                    self.json_buffer = ""
                if "```" in self.json_buffer:
                     if self.json_buffer.count("```") >= 2:
                         self.shield_active = False
            
        if self.content_buffer:
            clean = re.sub(r"</?think>", "", self.content_buffer)
            clean = clean.lstrip(": \n\r\t")
            if not re.search(r"(?:\{?\s*\"?(?:name|tool_name|search_web|get_weather|get_joke|rag_search)\"?|Action:|Call:|Tool:|Thought:|Assistant:)", clean):
                if self.is_thinking: yield {"type": "reasoning", "content": clean}
                else: yield {"type": "token", "content": clean}

class MetaAgentMock:
    def _repair_json(self, raw: str) -> str:
        repaired = re.sub(r"([{,])\s*([a-zA-Z_]\w*)\s*:", r'\1 "\2":', raw)
        repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
        repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        return repaired

    def _heuristically_extract_tool_call(self, text: str) -> dict[str, Any] | None:
        import json
        raw_json_match = re.search(r"(\{.*?(?:name|tool_name|search_web|get_weather|get_joke|rag_search).*?\})", text, re.DOTALL)
        if raw_json_match:
            try:
                repaired = self._repair_json(raw_json_match.group(1))
                data = json.loads(repaired)
                if "name" in data: return data
                if "tool_name" in data: return {"name": data["tool_name"], "arguments": data.get("arguments", {})}
            except Exception: pass

        action_match = re.search(r"(?:Action|Call|Tool):\s*\"?(search_web|get_weather|get_joke|rag_search)\"?", text, re.IGNORECASE)
        if action_match:
            tool = action_match.group(1).lower()
            input_match = re.search(r"(?:Action Input|Arguments|Args|Input|Query|Location):\s*\"?([^\"]+)\"?", text, re.IGNORECASE)
            if input_match:
                val = input_match.group(1).strip()
                return {"name": tool, "arguments": {"query" if tool == "search_web" else "location": val}}
        
        # 4. Last Chance: Super-aggressive scan for any known tool name
        known_tools = ["search_web", "get_weather", "get_joke", "rag_search"]
        for tool in known_tools:
            if tool in text:
                arg_match = re.search(r"(?:query|location|arguments|args|Action Input|Input):\s*(.+)", text, re.IGNORECASE)
                if arg_match:
                    val = arg_match.group(1).strip()
                    # Remove trailing markers if any
                    val = re.split(r"\s+(?:end|stop|Action:)\b", val, flags=re.IGNORECASE)[0]
                    # Strip outer braces/brackets
                    val = val.strip(' {}[]')
                    # Remove inner keys
                    val = re.sub(r'^(?:query|location|input)\s*[:=]\s*', '', val, flags=re.IGNORECASE)
                    # Final strip of quotes and whitespace
                    val = val.strip(' "\'')
                    return {"name": tool, "arguments": {"query" if tool == "search_web" else "location": val}}
        return None

async def mock_generator(chunks):
    for c in chunks: yield c

async def run_test(name, chunks, expected_tokens=None, expected_tool=None):
    print(f"--- Running Test: {name} ---")
    sanitizer = StreamSanitizer(mock_generator(chunks))
    tokens = []
    async for event in sanitizer:
        if event["type"] == "token": tokens.append(event["content"])
    
    full_text = "".join(tokens)
    print(f"Tokens: {full_text}")
    
    mock_agent = MetaAgentMock()
    # Simulate extraction from both tokens and buffer
    search_text = full_text + "\n" + sanitizer.json_buffer
    print(f"Search Text: [[{search_text}]]")
    tool = mock_agent._heuristically_extract_tool_call(search_text)
    print(f"Extracted Tool: {tool}")
    
    if expected_tokens:
        for t in expected_tokens: assert t in full_text
    if expected_tool:
        print(f"Expected Tool: {expected_tool}")
        assert tool and tool["name"] == expected_tool["name"]
        for k, v in expected_tool["arguments"].items():
            print(f"Comparing arg '{k}': extracted='{tool['arguments'].get(k)}', expected='{v}'")
            assert tool["arguments"][k] == v
    print(f"Test {name} PASSED!\n")

async def test_all():
    # 1. BitNet Pattern (Fragmented JSON)
    await run_test("BitNet Fragment", 
                  ["I will search. ", "name: search_web, ", "arguments: {query: \"kolkata news\"}"],
                  expected_tool={"name": "search_web", "arguments": {"query": "kolkata news"}})
    
    # 2. Llama 3 Pattern (Action/Action Input)
    await run_test("Llama 3 Action",
                  ["I need more info. \nAction: search_web\nAction Input: \"petrol price today\""],
                  expected_tokens=["I need more info"], # Action should be shielded
                  expected_tool={"name": "search_web", "arguments": {"query": "petrol price today"}})
    
    # 3. Gemma Pattern (Thought/Action)
    await run_test("Gemma 2 Thought",
                  ["Thought: I should check weather.\nAction: get_weather\nLocation: Delhi"],
                  expected_tool={"name": "get_weather", "arguments": {"location": "Delhi"}})

    # 4. Shielding Test
    await run_test("Shielding Check",
                  ["Normal text ", "Thought: hide me ", "Action: search_web", " end."],
                  expected_tokens=["Normal text", "end"])
    # "Thought" and "Action" should NOT be in Tokens

if __name__ == "__main__":
    asyncio.run(test_all())
