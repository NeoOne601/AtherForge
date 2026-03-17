
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modules.search.engines import DuckDuckGoEngine, StartpageEngine, BraveEngine

async def debug_engines():
    engines = [
        DuckDuckGoEngine(),
        StartpageEngine(),
        BraveEngine()
    ]
    
    query = "current price of gold in USD"
    print(f"DEBUG: Querying {query}\n")
    
    for engine in engines:
        print(f"--- Testing {engine.name} ---")
        try:
            results = await engine.search(query)
            print(f"  Results found: {len(results)}")
            if results:
                for r in results[:2]:
                    print(f"  - {r.title} ({r.url})")
            else:
                print("  WARNING: No results returned.")
        except Exception as e:
            print(f"  ERROR: {e}")
        print("\n")

if __name__ == "__main__":
    asyncio.run(debug_engines())
