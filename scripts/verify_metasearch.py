
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modules.search.aggregator import MetasearchAggregator

async def test_metasearch():
    print("Testing Metasearch Aggregator...")
    aggregator = MetasearchAggregator()
    
    query = "current gas prices in Kolkata"
    print(f"Query: {query}")
    
    results = await aggregator.aggregate(query)
    
    print(f"\nFound {len(results)} results:")
    for i, r in enumerate(results):
        print(f"{i+1}. [{r.engine}] {r.title} ({r.score:.2f})")
        print(f"   URL: {r.url}")
        print(f"   Snippet: {r.snippet[:100]}...")
        print("-" * 20)

    assert len(results) > 0, "Should find at least some results"
    
    # Check for deduplication (should not have duplicate URLs essentially)
    urls = [r.url for r in results]
    assert len(urls) == len(set(urls)), "Results should be deduplicated by URL"
    
    # Check if multiple engines contributed (heuristic)
    engines = set(r.engine for r in results)
    print(f"\nEngines contributing: {engines}")
    
    print("\nMetasearch verification PASSED!")

if __name__ == "__main__":
    asyncio.run(test_metasearch())
