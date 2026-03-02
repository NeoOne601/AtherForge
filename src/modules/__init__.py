# AetherForge v1.0 — src/modules/__init__.py
# Module registry — all 5 AetherForge LangGraph module graphs
from src.modules.ragforge.graph import build_ragforge_graph
from src.modules.localbuddy.graph import build_localbuddy_graph
from src.modules.watchtower.graph import build_watchtower_graph
from src.modules.streamsync.graph import build_streamsync_graph
from src.modules.tunelab.graph import build_tunelab_graph

__all__ = [
    "build_ragforge_graph",
    "build_localbuddy_graph",
    "build_watchtower_graph",
    "build_streamsync_graph",
    "build_tunelab_graph",
]
