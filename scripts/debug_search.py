
import asyncio
from src.modules.search.aggregator import MetasearchAggregator

async def test_search():
    agg = MetasearchAggregator()
    query = "current weather in Kolkata"
    print(f"Testing search for: {query}")
    results = await agg.aggregate(query)
    
    if not results:
        print("No results found.")
    else:
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r.engine}: {r.title} ({r.url})")
            print(f"    Snippet: {r.snippet[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_search())
