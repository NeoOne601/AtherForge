"""
AetherForge Analytics Module — File Generation Engine
=====================================================
Provides create_visual and analyze_data tools that generate
downloadable files (PNG charts, CSV reports) from document data.

Generated files are saved to data/generated/ and served via
the /api/v1/generated/{filename} endpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import structlog

from src.modules.base import BaseModule

logger = structlog.get_logger("aetherforge.modules.analytics")

# Directory for generated output files
GENERATED_DIR = Path(os.environ.get("DATA_DIR", "data")) / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


class AnalyticsModule(BaseModule):
    """Module for data visualization and analysis tools."""

    def __init__(self) -> None:
        super().__init__(name="analytics")

    async def initialize(self) -> None:
        pass

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "create_visual",
                "description": (
                    "Creates a chart or diagram and saves it as a downloadable PNG image. "
                    "Use this when the user asks for a chart, plot, graph, diagram, "
                    "flowchart, or any visual representation of data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "description": (
                                "Type of chart: 'bar', 'line', 'pie', 'scatter', "
                                "'hbar', 'area', 'table', or 'flowchart'"
                            ),
                        },
                        "title": {
                            "type": "string",
                            "description": "Title of the chart",
                        },
                        "labels": {
                            "type": "array",
                            "description": "Labels for data points (x-axis or categories)",
                            "items": {"type": "string"},
                        },
                        "values": {
                            "type": "array",
                            "description": "Numeric values for data points (y-axis or sizes)",
                            "items": {"type": "number"},
                        },
                        "x_label": {
                            "type": "string",
                            "description": "Label for x-axis (optional)",
                        },
                        "y_label": {
                            "type": "string",
                            "description": "Label for y-axis (optional)",
                        },
                        "series_names": {
                            "type": "array",
                            "description": "Names for multiple data series (optional)",
                            "items": {"type": "string"},
                        },
                        "multi_values": {
                            "type": "array",
                            "description": "Multiple data series, each an array of numbers (optional)",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                        },
                    },
                    "required": ["chart_type", "title", "labels", "values"],
                },
            },
            {
                "name": "analyze_data",
                "description": (
                    "Analyzes data from retrieved documents and generates a downloadable "
                    "CSV or Markdown report. Use this when the user asks for data analysis, "
                    "summaries, tabular output, or reports."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_title": {
                            "type": "string",
                            "description": "Title of the analysis report",
                        },
                        "format": {
                            "type": "string",
                            "description": "'csv' or 'markdown'",
                        },
                        "headers": {
                            "type": "array",
                            "description": "Column headers for the data table",
                            "items": {"type": "string"},
                        },
                        "rows": {
                            "type": "array",
                            "description": "Rows of data, each row is an array of values",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "summary": {
                            "type": "string",
                            "description": "Optional text summary of findings",
                        },
                    },
                    "required": ["report_title", "headers", "rows"],
                },
            },
        ]

    def register_tools(self) -> None:
        from src.core.tool_registry import tool_registry

        for defn in self.get_tool_definitions():
            handler = {
                "create_visual": self._handle_create_visual,
                "analyze_data": self._handle_analyze_data,
            }.get(defn["name"])
            if handler:
                tool_registry.register_tool(defn, handler)

    # ─────────────────────────────────────────────────────────────
    # Tool Handlers
    # ─────────────────────────────────────────────────────────────

    def _handle_create_visual(self, args: dict[str, Any]) -> str:
        """Generate a chart image using matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            return "[Error] matplotlib is not installed. Run: pip install matplotlib"

        chart_type = str(args.get("chart_type", "bar")).lower()
        title = str(args.get("title", "Chart"))
        labels = args.get("labels", [])
        values = args.get("values", [])
        x_label = args.get("x_label", "")
        y_label = args.get("y_label", "")
        multi_values = args.get("multi_values", None)
        series_names = args.get("series_names", None)

        # Ensure values are numeric
        try:
            values = [float(v) for v in values]
        except (ValueError, TypeError):
            values = list(range(len(labels)))

        # Generate unique filename
        ts = int(time.time())
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:30]
        filename = f"chart_{slug}_{ts}.png"
        filepath = GENERATED_DIR / filename

        # Premium dark theme
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color palette
        colors = [
            "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
            "#818cf8", "#4f46e5", "#7c3aed", "#5b21b6",
        ]

        if chart_type == "bar":
            bars = ax.bar(labels, values, color=colors[:len(values)],
                         edgecolor="white", linewidth=0.5, alpha=0.9)
            for bar_item, val in zip(bars, values):
                ax.text(bar_item.get_x() + bar_item.get_width() / 2, bar_item.get_height() + 0.5,
                       f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="white")

        elif chart_type == "hbar":
            bars = ax.barh(labels, values, color=colors[:len(values)],
                          edgecolor="white", linewidth=0.5, alpha=0.9)

        elif chart_type == "line":
            if multi_values and series_names:
                for i, (series, name) in enumerate(zip(multi_values, series_names)):
                    try:
                        series_floats = [float(s) for s in series]
                    except (ValueError, TypeError):
                        continue
                    ax.plot(labels[:len(series_floats)], series_floats,
                           marker="o", label=name, color=colors[i % len(colors)], linewidth=2)
                ax.legend()
            else:
                ax.plot(labels, values, marker="o", color=colors[0], linewidth=2)
            ax.fill_between(range(len(values)), values, alpha=0.1, color=colors[0])

        elif chart_type == "scatter":
            ax.scatter(range(len(values)), values, c=colors[:len(values)],
                      s=100, edgecolors="white", zorder=5)
            for i, (lbl, val) in enumerate(zip(labels, values)):
                ax.annotate(lbl, (i, val), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=8, color="white")

        elif chart_type == "pie":
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, colors=colors[:len(values)],
                autopct="%1.1f%%", startangle=90,
                textprops={"color": "white", "fontsize": 10},
            )
            for autotext in autotexts:
                autotext.set_fontweight("bold")

        elif chart_type == "area":
            ax.fill_between(range(len(values)), values, alpha=0.4, color=colors[0])
            ax.plot(range(len(values)), values, color=colors[0], linewidth=2)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")

        elif chart_type == "flowchart":
            # Simple box-and-arrow flowchart
            ax.set_xlim(0, 10)
            ax.set_ylim(0, len(labels) * 2 + 1)
            ax.axis("off")
            for i, label in enumerate(labels):
                y = len(labels) * 2 - i * 2
                box = mpatches.FancyBboxPatch(
                    (2, y - 0.4), 6, 0.8,
                    boxstyle="round,pad=0.3", facecolor=colors[i % len(colors)],
                    edgecolor="white", linewidth=1.5,
                )
                ax.add_patch(box)
                ax.text(5, y, label, ha="center", va="center",
                       fontsize=11, color="white", fontweight="bold")
                if i < len(labels) - 1:
                    ax.annotate(
                        "", xy=(5, y - 0.5), xytext=(5, y - 1.5),
                        arrowprops=dict(arrowstyle="->", color="white", lw=2),
                    )

        elif chart_type == "table":
            ax.axis("off")
            if values:
                table_data = [[lbl, f"{val:.2f}"] for lbl, val in zip(labels, values)]
                table = ax.table(
                    cellText=table_data,
                    colLabels=["Item", "Value"],
                    cellLoc="center",
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                table.scale(1.2, 1.5)
                for key, cell in table.get_celld().items():
                    if key[0] == 0:
                        cell.set_facecolor(colors[0])
                        cell.set_text_props(color="white", fontweight="bold")
                    else:
                        cell.set_facecolor("#1e1e2e")
                        cell.set_text_props(color="white")
                    cell.set_edgecolor("#555")
        else:
            # Fallback to bar
            ax.bar(labels, values, color=colors[:len(values)])

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="white")
        if x_label and chart_type not in ("pie", "flowchart", "table"):
            ax.set_xlabel(x_label, fontsize=12)
        if y_label and chart_type not in ("pie", "flowchart", "table"):
            ax.set_ylabel(y_label, fontsize=12)

        if chart_type not in ("pie", "flowchart", "table"):
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.2)

        plt.tight_layout()
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Chart generated", filename=filename, chart_type=chart_type)
        return f"[attachment:{filename}]"

    def _handle_analyze_data(self, args: dict[str, Any]) -> str:
        """Generate a data report as CSV or Markdown file."""
        report_title = str(args.get("report_title", "Analysis Report"))
        fmt = str(args.get("format", "markdown")).lower()
        headers = args.get("headers", [])
        rows = args.get("rows", [])
        summary = args.get("summary", "")

        ts = int(time.time())
        slug = re.sub(r"[^a-z0-9]+", "_", report_title.lower())[:30]

        if fmt == "csv":
            filename = f"report_{slug}_{ts}.csv"
            filepath = GENERATED_DIR / filename

            lines = [",".join(str(h) for h in headers)]
            for row in rows:
                lines.append(",".join(str(cell) for cell in row))
            filepath.write_text("\n".join(lines), encoding="utf-8")

        else:
            # Markdown report
            filename = f"report_{slug}_{ts}.md"
            filepath = GENERATED_DIR / filename

            content = f"# {report_title}\n\n"
            if summary:
                content += f"{summary}\n\n"

            # Build markdown table
            if headers:
                content += "| " + " | ".join(str(h) for h in headers) + " |\n"
                content += "| " + " | ".join("---" for _ in headers) + " |\n"
                for row in rows:
                    content += "| " + " | ".join(str(cell) for cell in row) + " |\n"
                content += "\n"

            content += f"\n---\n*Generated by AetherForge Analytics at {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
            filepath.write_text(content, encoding="utf-8")

        logger.info("Report generated", filename=filename, format=fmt)
        return f"[attachment:{filename}]"
