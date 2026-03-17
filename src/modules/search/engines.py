import asyncio
import warnings
from typing import List, Dict, Any
import httpx
from bs4 import BeautifulSoup
import structlog
from src.modules.search.base import BaseSearchEngine, SearchResult

logger = structlog.get_logger("aetherforge.search.engines")

class DuckDuckGoEngine(BaseSearchEngine):
    """DuckDuckGo Search Engine."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(name="duckduckgo", weight=weight)

    async def search(self, query: str) -> List[SearchResult]:
        try:
            # Suppress RuntimeWarning about package rename
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*renamed to.*ddgs.*")
                try:
                    from duckduckgo_search import DDGS
                except ImportError:
                    try:
                        from ddgs import DDGS
                    except ImportError:
                        return []
                
                # DDGS is blocking, so we run in a thread
                def _sync_search():
                    with DDGS() as ddgs:
                        return list(ddgs.text(query, max_results=8))
                
                results = await asyncio.to_thread(_sync_search)
                
                return [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", r.get("snippet", "")),
                        engine=self.name,
                        score=self.weight
                    )
                    for r in results if r.get("href")
                ]
        except Exception as e:
            logger.warning("DuckDuckGo engine failed", error=str(e))
            return []

class StartpageEngine(BaseSearchEngine):
    """Startpage Search Engine."""
    
    def __init__(self, weight: float = 0.9):
        super().__init__(name="startpage", weight=weight)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def search(self, query: str) -> List[SearchResult]:
        try:
            async with httpx.AsyncClient(headers=self.headers, follow_redirects=True, timeout=10.0) as client:
                # Direct search attempt
                resp = await client.get("https://www.startpage.com/do/search", params={"query": query})
                
                if resp.status_code != 200:
                    return []
                
                soup = BeautifulSoup(resp.text, "html.parser")
                # Improved Startpage selectors
                containers = soup.select(".w-gl__result") or soup.select(".result") or soup.select(".div.clk") or soup.select("div.result")
                
                results = []
                for result in containers[:10]:
                    title_elem = result.select_one(".w-gl__result-title") or result.select_one("h3") or result.select_one(".title")
                    link_elem = result.select_one("a.w-gl__result-link") or result.select_one("a")
                    snippet_elem = result.select_one(".w-gl__description") or result.select_one(".description") or result.select_one(".content") or result.select_one(".st")
                    
                    if title_elem and link_elem:
                        url = link_elem.get("href", "")
                        if url.startswith("/"):
                             url = "https://www.startpage.com" + url
                        
                        if "startpage.com" in url and ("rdr" in url or "sp/search" in url):
                             continue
                             
                        results.append(SearchResult(
                            title=title_elem.get_text(strip=True),
                            url=url,
                            snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                            engine=self.name,
                            score=self.weight
                        ))
                return results
        except Exception as e:
            logger.warning("Startpage engine failed", error=str(e))
            return []

class BraveEngine(BaseSearchEngine):
    """Brave Search Engine (Scraper)."""
    
    def __init__(self, weight: float = 1.2):
        super().__init__(name="brave", weight=weight)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept-Encoding": "gzip, deflate", 
        }

    async def search(self, query: str) -> List[SearchResult]:
        try:
            async with httpx.AsyncClient(headers=self.headers, follow_redirects=True, timeout=10.0) as client:
                resp = await client.get("https://search.brave.com/search", params={"q": query})
                if resp.status_code != 200:
                    return []
                
                soup = BeautifulSoup(resp.text, "html.parser")
                containers = soup.select(".snippet") or soup.select("div[data-type='web']") or soup.select(".result")
                
                results = []
                for result in containers[:10]:
                    title_elem = result.select_one(".title") or result.select_one(".result-header") or result.select_one("h2")
                    link_elem = result.select_one("a")
                    snippet_elem = result.select_one(".snippet-content") or result.select_one(".result-content") or result.select_one(".snippet-description")
                    
                    if title_elem and link_elem:
                        url = link_elem.get("href", "")
                        if not url.startswith("http"): continue
                        
                        results.append(SearchResult(
                            title=title_elem.get_text(strip=True),
                            url=url,
                            snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                            engine=self.name,
                            score=self.weight
                        ))
                return results
        except Exception as e:
            logger.warning("Brave engine failed", error=str(e))
            return []

class WikipediaEngine(BaseSearchEngine):
    """Wikipedia Search Engine for high-quality factual context."""
    
    def __init__(self, weight: float = 1.3):
        super().__init__(name="wikipedia", weight=weight)

    async def search(self, query: str) -> List[SearchResult]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Search API
                resp = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json"
                    }
                )
                data = resp.json()
                search_recs = data.get("query", {}).get("search", [])
                
                results = []
                for r in search_recs[:3]:
                    title = r.get("title", "")
                    page_id = r.get("pageid")
                    snippet = BeautifulSoup(r.get("snippet", ""), "html.parser").get_text()
                    
                    if title:
                        results.append(SearchResult(
                            title=title,
                            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            snippet=f"[Wikipedia] {snippet}...",
                            engine=self.name,
                            score=self.weight
                        ))
                return results
        except Exception as e:
            logger.warning("Wikipedia engine failed", error=str(e))
            return []
