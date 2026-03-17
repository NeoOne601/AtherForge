import asyncio
from typing import List, Dict, Any
import structlog
from src.modules.search.base import SearchResult
from src.modules.search.engines import DuckDuckGoEngine, StartpageEngine, BraveEngine, WikipediaEngine

logger = structlog.get_logger("aetherforge.search.aggregator")

class MetasearchAggregator:
    """
    Aggregates results from multiple search engines, deduplicates, and ranks them.
    Inspired by SearXNG's architectural pattern.
    """
    
    def __init__(self):
        self.engines = [
            DuckDuckGoEngine(weight=1.0),
            StartpageEngine(weight=0.9),
            BraveEngine(weight=1.2),
            WikipediaEngine(weight=1.3)
        ]

    async def aggregate(self, query: str, timeout: float = 8.0) -> List[SearchResult]:
        """
        Execute metasearch aggregator loop.
        1. Parallel Dispatch
        2. Deduplication (Merge on URL)
        3. Multi-Factor Ranking
        """
        logger.info("Starting metasearch aggregation", query=query)
        
        # 1. Parallel Dispatch using asyncio.gather with timeout
        tasks = [asyncio.wait_for(engine.search(query), timeout=timeout) for engine in self.engines]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_results: List[SearchResult] = []
        for engine_results in results_nested:
            if isinstance(engine_results, list):
                all_results.extend(engine_results)
            elif isinstance(engine_results, asyncio.TimeoutError):
                logger.warning("Engine search timed out", query=query)
            elif isinstance(engine_results, Exception):
                logger.error("Engine sub-agent failed", error=str(engine_results))

        if not all_results:
            return []

        # 2. Deduplication (Merge results with same URL)
        unique_results: Dict[str, SearchResult] = {}
        for res in all_results:
            # Normalize URL to remove tracking params/fragments for better merging
            clean_url = self._normalize_url(res.url)
            
            if clean_url in unique_results:
                existing = unique_results[clean_url]
                # Merge: Boost score if multiple engines found it
                existing.score += res.score * 0.5
                # Keep the longest snippet if available
                if len(res.snippet) > len(existing.snippet):
                    existing.snippet = res.snippet
            else:
                res.url = clean_url
                unique_results[clean_url] = res

        # 3. Multi-Factor Ranking
        # Factors: Base engine weight + Multi-discovery boost + Relevance heuristic
        ranked_results = sorted(
            unique_results.values(), 
            key=lambda x: x.score, 
            reverse=True
        )

        return ranked_results[:10]  # Return top 10 best results

    def _normalize_url(self, url: str) -> str:
        """Strip tracking parameters and trailing slashes for deduplication."""
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            # Remove query params that are likely tracking (heuristic)
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', '', ''))
        except Exception:
            return url
