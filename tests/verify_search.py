
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.modules.core_tools import SearchSubAgent
import structlog

# Setup logging
structlog.configure()

def test_search():
    agent = SearchSubAgent()
    print("--- Testing DuckDuckGo Search ---")
    results = agent.search("Python programming news")
    print(f"Status: {results['status']}")
    print(f"Engine: {results['engine']}")
    if results['status'] == 'success':
        for i, r in enumerate(results['results'][:2], 1):
            print(f"{i}. {r['title']} - {r['url']}")
    
    print("\n--- Testing Fallback (Simulating DDG Failure) ---")
    # Patch _duckduckgo_search to return empty
    agent._duckduckgo_search = lambda q: []
    results_fall = agent.search("OpenAI Sora release date")
    print(f"Status: {results_fall['status']}")
    print(f"Engine: {results_fall['engine']}")
    if results_fall['status'] == 'success':
        for i, r in enumerate(results_fall['results'][:2], 1):
            print(f"{i}. {r['title']} - {r['url']}")
    else:
        print("Fallback failed or returned no results.")

if __name__ == "__main__":
    test_search()
