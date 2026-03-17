from __future__ import annotations

import random
import warnings
from typing import Any

import structlog

# Suppress the RuntimeWarning about duckduckgo_search package rename globally for this module
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*renamed to.*ddgs.*")

from src.config import AetherForgeSettings
from src.modules.base import BaseModule

logger = structlog.get_logger("aetherforge.modules.core")

# Curated AI jokes so the model never needs to invent one.
_AI_JOKES = [
    "Why did the neural network go to therapy? It had too many unresolved layers.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "An AI walked into a bar. The bartender said, "
    "'We don't serve your kind here.' The AI replied, "
    "'But I'm fully aligned!'",
    "How does an AI break up with you? 'It's not you, it's my training data.'",
    "Why was the transformer model so popular? Because it always paid attention.",
    "What did GPT say to BERT? 'You only look once? That's so YOLO.'",
]



from src.modules.search.aggregator import MetasearchAggregator



class CoreModule(BaseModule):
    """Module for system-level tools (Web Search, Weather, Jokes)."""

    def __init__(self, settings: AetherForgeSettings) -> None:
        super().__init__(name="core")
        self.settings = settings
        self._aggregator = MetasearchAggregator()

    @property
    def system_prompt_extension(self) -> str:
        return (
            "\n\nYou have access to system-level tools for web search, weather, and "
            "humor. Use them when appropriate to provide real-time data or a helpful tone."
        )

    async def initialize(self) -> None:
        pass

    async def process(self, payload: dict[str, Any], state: Any = None) -> dict[str, Any]:
        """
        Execute core system logic.
        For now, this is a pass-through to the tool execution loop.
        """
        return {
            "content": "[CoreModule] Protocol initialized. Awaiting tool dispatch.",
            "metadata": {},
            "causal_edges": [
                {"source": "core_start", "target": "ready", "label": "Protocol Initialized"}
            ],
        }
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_web",
                "description": (
                    "Searches the live internet using a metasearch aggregator (DDG, Startpage, Brave). "
                    "Provides high-availability results and cross-engine deduplication."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": ("The search query to look up on the internet."),
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_weather",
                "description": (
                    "Gets the current weather for a specific city "
                    "or location using SI units (°C, km/h)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": ("City and country or a well-known place name."),
                        },
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_joke",
                "description": (
                    "Returns a fun joke about AI, programming, "
                    "or technology. Use when the user asks for "
                    "a joke, humor, or something funny."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        state: Any | None = None,
    ) -> str:
        if name == "search_web":
            # Search is async, but execute_tool is sync.
            # Since we are already in a dedicated thread (via MetaAgent), 
            # we can run a new event loop.
            import asyncio
            query = args.get("query", "")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Add a strict safety timeout at the bridge level
                results = loop.run_until_complete(
                    asyncio.wait_for(self._aggregator.aggregate(query), timeout=15.0)
                )
                return self._format_search_results(results)
            except asyncio.TimeoutError:
                logger.error("Metasearch bridge timeout", query=query)
                return "Search failed: The search took too long to respond. Please try again or simplify your query."
            except Exception as e:
                logger.error("Metasearch bridge failure", error=str(e))
                return f"Search failed: {str(e)}"
            finally:
                loop.close()
        elif name == "get_weather":
            loc = args.get("location", "")
            if not loc and state and hasattr(state, "system_location"):
                loc = state.system_location
            return self._get_weather(loc)
        elif name == "get_joke":
            return self._get_joke()
        return f"Error: Tool '{name}' not found in Core module."

    def _format_search_results(self, results: list) -> str:
        if not results:
            return "No results found on the internet."
        
        chunks = []
        for i, r in enumerate(results, 1):
            chunks.append(
                f"[{i}] Source: {r.url} (Engine: {r.engine})\n"
                f"Title: {r.title}\n"
                f"Snippet: {r.snippet}"
            )
        
        header = f"Metasearch Results (Aggregated and Numbered for verification):\n\n"
        return header + "\n\n".join(chunks)

    def _get_joke(self) -> str:
        return random.choice(_AI_JOKES)

    def _search_web(self, query: str) -> str:
        # This is now handled inside execute_tool via the bridge
        return "DEPRECATED: Use execute_tool directly"

    def _get_weather(self, location: str) -> str:
        import httpx

        try:
            with httpx.Client(timeout=5.0) as client:
                geo_resp = client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={
                        "name": location,
                        "count": 1,
                        "language": "en",
                        "format": "json",
                    },
                )
                geo = geo_resp.json()
                results = geo.get("results") or []
                if not results:
                    return f"Weather lookup failed: could not resolve location '{location}'."
                place = results[0]
                lat = place.get("latitude")
                lon = place.get("longitude")
                resolved_name = (f"{place.get('name', '')}, {place.get('country', '')}").strip(", ")
                wx_resp = client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": (
                            "temperature_2m,"
                            "relative_humidity_2m,"
                            "apparent_temperature,"
                            "precipitation,"
                            "weather_code,"
                            "wind_speed_10m"
                        ),
                        "temperature_unit": "celsius",
                        "wind_speed_unit": "kmh",
                        "timezone": "auto",
                    },
                )
                wx = wx_resp.json()
                current = wx.get("current") or {}
                return (
                    f"Resolved location: {resolved_name} "
                    f"(lat {lat}, lon {lon})\n"
                    f"Temperature: "
                    f"{current.get('temperature_2m')} °C "
                    f"(feels like "
                    f"{current.get('apparent_temperature')} °C)\n"
                    f"Relative humidity: "
                    f"{current.get('relative_humidity_2m')}%\n"
                    f"Wind speed: "
                    f"{current.get('wind_speed_10m')} km/h\n"
                )
        except Exception as e:
            return f"Weather lookup failed: {e}"
