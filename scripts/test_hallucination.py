
import asyncio
import json
import websockets

async def test_hallucination():
    uri = "ws://127.0.0.1:8765/api/v1/ws/ui-session-test"
    async with websockets.connect(uri) as websocket:
        # 1. Send query
        payload = {
            "message": "What is the current weather in Kolkata? Provide details including temperature and wind.",
            "module": "localbuddy",
            "context": {"web_search_enabled": True}
        }
        await websocket.send(json.dumps(payload))
        
        print("Sent query. Waiting for response...")
        
        # 2. Receive events
        full_response = ""
        while True:
            try:
                msg = await websocket.recv()
                data = json.loads(msg)
                
                if data["type"] == "token":
                    full_response += data["content"]
                    print(data["content"], end="", flush=True)
                elif data["type"] == "reasoning":
                    print(f"\n[Reasoning] {data['content']}", end="", flush=True)
                elif data["type"] == "tool_start":
                    print(f"\n[Tool] {data['name']} {data['args']}")
                elif data["type"] == "done":
                    print("\n[Done]")
                    break
            except Exception as e:
                print(f"\n[Error] {e}")
                break
        
        # 3. Validation
        print("\n--- FINAL ANALYSIS ---")
        if "[Source" in full_response:
            print("SUCCESS: Citations found.")
        else:
            print("FAILURE: No citations found. Auditor might be bypassed or failing.")
            
        if "Kolkata" in full_response:
            print("SUCCESS: Relevant content found.")
        else:
            print("FAILURE: Kolkata not mentioned in final response.")

if __name__ == "__main__":
    asyncio.run(test_hallucination())
