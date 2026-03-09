import asyncio
import structlog
import json
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import feedparser
import requests
from bs4 import BeautifulSoup

from src.main import get_state, app
from src.modules.ragforge_indexer import index_document
from src.modules.streamsync.graph import emit_event

logger = structlog.get_logger("aetherforge.streamsync.rss")

class RSSFeeder:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.state_file = self.data_dir / "streamsync_rss_state.json"
        self.seen_urls = set()
        self.load_state()
        
    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.seen_urls = set(data.get("seen_urls", []))
            except Exception as e:
                logger.error("Failed to load RSS state: %s", e)
                
    def save_state(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump({"seen_urls": list(self.seen_urls)}, f)
        except Exception as e:
            logger.error("Failed to save RSS state: %s", e)

    def extract_text_from_url(self, url: str) -> str:
        """Scrape main article text using BeautifulSoup."""
        headers = {"User-Agent": "AetherForge-StreamSync/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Kill all script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
            
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    async def ingest_article(self, entry: feedparser.FeedParserDict, state) -> bool:
        title = entry.get("title", "Untitled News")
        link = entry.get("link", "")
        
        if not link or link in self.seen_urls:
            return False
            
        logger.info("StreamSync RSS ingesting new article: %s", title)
        
        try:
            # 1. Scrape the raw text
            text = await asyncio.to_thread(self.extract_text_from_url, link)
            if len(text) < 500:
                logger.warning("Article text too short, skipping %s", link)
                self.seen_urls.add(link)
                self.save_state()
                return False
                
            # 2. Write to a temporary Markdown file
            safe_title = "".join([c if c.isalnum() else "_" for c in title[:50]])
            temp_path = self.data_dir / f"rss_{safe_title}.md"
            
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\nURL: {link}\nDate: {datetime.now().isoformat()}\n\n{text}")
                
            # 3. Index into RAGForge
            if state.vector_store and state.sparse_index:
                result = await asyncio.to_thread(
                    index_document, temp_path, state.vector_store, state.sparse_index
                )
                chunks_added = result.get("chunks_added", 0) if isinstance(result, dict) else int(result)
                
                # 4. Cleanup temp file
                temp_path.unlink(missing_ok=True)
                
                # 5. Emit StreamSync UI Event
                emit_event(
                    event_type="rss_article_ingested",
                    source="RSSFeeder",
                    payload={
                        "title": title,
                        "url": link,
                        "chunks": chunks_added,
                    }
                )
                # Mark as seen
                self.seen_urls.add(link)
                self.save_state()
                return True
            else:
                logger.error("Database not ready to index RSS article")
                return False
                
        except Exception as e:
            logger.error("Failed to ingest RSS article %s: %s", link, e)
            emit_event(
                event_type="rss_article_failed",
                source="RSSFeeder",
                payload={"title": title, "url": link, "error": str(e)}
            )
            return False

async def rss_poller_task(app):
    """Background task running every 30 minutes to fetch feeds."""
    logger.info("StreamSync RSS Poller background task started.")
    
    # Wait for startup sequences to finish
    await asyncio.sleep(10)
    
    state = get_state(app)
    feeder = RSSFeeder(state.settings.data_dir)
    
    while True:
        # Load active RSS feeds from state (added via UI)
        feeds = getattr(state, "streamsync_rss_feeds", [])
        
        if not feeds:
            await asyncio.sleep(60)
            continue
            
        logger.info("StreamSync RSS polling %d feeds", len(feeds))
        
        new_articles = 0
        for feed_url in feeds:
            try:
                parsed = await asyncio.to_thread(feedparser.parse, feed_url)
                # Only process the 5 most recent articles to avoid blowing up the index on first run
                for entry in parsed.entries[:5]:
                    success = await feeder.ingest_article(entry, state)
                    if success:
                        new_articles += 1
                        # Throttle to avoid rate limits
                        await asyncio.sleep(2)
            except Exception as e:
                logger.error("Failed to fetch RSS feed %s: %s", feed_url, e)
                
        if new_articles > 0:
            logger.info("StreamSync RSS polling complete. %d new articles ingested.", new_articles)
            
        # Poll every 30 minutes
        await asyncio.sleep(1800)
