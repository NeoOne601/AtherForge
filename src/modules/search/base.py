from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from pydantic import BaseModel

class SearchResult(BaseModel):
    """Standardized search result schema."""
    title: str
    url: str
    snippet: str
    engine: str
    score: float = 1.0
    metadata: Dict[str, Any] = {}

class BaseSearchEngine(ABC):
    """Abstract base class for all search engine sub-agents."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    async def search(self, query: str) -> List[SearchResult]:
        """Execute search and return standardized results."""
        pass
