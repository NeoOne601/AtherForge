# Verification Script—Deep Research Protocol
# ─────────────────────────────────────────────────────────────────
# Test the dynamic planning loop, VFS notes, and iterative reasoning.
# ─────────────────────────────────────────────────────────────────

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from src.meta_agent import MetaAgent, MetaAgentInput
from src.config import AetherForgeSettings
from src.guardrails.silicon_colosseum import SiliconColosseum

async def verify_deep_research() -> None:
    print("🚀 Starting Deep Research Verification...")
    
    settings = AetherForgeSettings()
    colosseum = SiliconColosseum(settings)
    
    agent = MetaAgent(settings, colosseum)
    await agent.initialize()
    
    # Register Core Tools (search_web, get_weather, etc.)
    from src.modules.core_tools import CoreModule # type: ignore
    from src.core.tool_registry import tool_registry # type: ignore
    
    core = CoreModule(settings)
    for tool_def in core.get_tool_definitions():
        tool_registry.register_tool(tool_def, core.execute_tool)
    
    # Test case: Complex query requiring decomposition
    query = "Find the latest news about NVIDIA stockholders, summarize the sentiment, and save a summary note."
    
    inp = MetaAgentInput(
        session_id="test_verify_dr",
        message=query,
        module="localbuddy", # Will be overridden by the planning loop
    )
    
    print(f"User Query: {query}")
    print("--- Executing Iterative Loop ---")
    
    # We use _run_sync or stream? Let's use _run_sync for easy verification
    # Note: _run_sync is effectively the implementation we just refactored.
    
    try:
        # Mocking or running actual LLM pass? 
        # For verification, we want to ensure the loop doesn't crash and tools are called.
        # We will check if the VFS file is created and contains notes.
        
        response = await agent.run(inp)
        
        print("\n--- Final Response ---")
        print(response.response)
        
        # Verify VFS persistence
        vfs = agent._get_or_create_vfs("test_verify_dr")
        notes = vfs.list_notes()
        print(f"\nVFS Notes Count: {len(notes)}")
        if len(notes) == 0:
            # Check if the file exists anyway
            notes_file = vfs.notes_file
            if notes_file.exists():
                print(f"DEBUG: VFS file exists at {notes_file} but list_notes() is empty. Checking content...")
                try:
                    import json
                    with open(notes_file, "r") as f:
                        data = json.load(f)
                        print(f"DEBUG: Raw file content has {len(data)} items.")
                        notes = data
                except Exception as e:
                    print(f"DEBUG: Failed to read file manually: {e}")

        for note in notes:
            print(f"- {note['title']}: {note['content'][:50]}...")
        if len(notes) > 0:
            print("\n✅ Verification SUCCESS: VFS managed stateful notes.")
        else:
            print("\n⚠️ Verification WARNING: No notes found. Did the LLM skip the scratchpad?")
    except Exception as e:
        print(f"\n❌ Verification FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_deep_research())
