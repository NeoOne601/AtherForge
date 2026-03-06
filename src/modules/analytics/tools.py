# AetherForge v1.0 — src/modules/analytics/tools.py
# ─────────────────────────────────────────────────────────────────
# DataVault Analytics Module
#
# Provides tools for querying DataFrames and generating charts.
# ─────────────────────────────────────────────────────────────────
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("aetherforge.analytics.tools")

def get_tools() -> List[Dict[str, Any]]:
    """Return the list of tools available in the analytics module."""
    return [
        {
            "name": "analyze_data",
            "description": "Performs data analysis on an indexed CSV or Excel file. Use this for aggregations, filtering, and summary stats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "The filename (e.g., 'sales.xlsx') to analyze."},
                    "operation": {"type": "string", "description": "The operation to perform: 'summary', 'group_by', or 'filter'."},
                    "column": {"type": "string", "description": "Target column for group_by or aggregation."},
                    "agg_func": {"type": "string", "description": "Aggregation function: 'sum', 'mean', 'count'. Only for group_by."}
                },
                "required": ["source", "operation"]
            }
        },
        {
            "name": "create_visual",
            "description": "Generates a chart (bar, line, scatter) from provided data points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter"], "description": "Type of chart."},
                    "title": {"type": "string", "description": "Title of the chart."},
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "X-axis labels or category names."},
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Y-axis values or data points."}
                },
                "required": ["chart_type", "title", "labels", "values"]
            }
        }
    ]

def execute_tool(name: str, args: Dict[str, Any], state: Any = None) -> str:
    """Execute the requested analytics tool."""
    try:
        if name == "analyze_data":
            return _analyze_data(args, state)
        elif name == "create_visual":
            return _create_visual(args)
        else:
            return f"Tool '{name}' not found in analytics module."
    except Exception as e:
        logger.error("Analytics tool execution failed: %s", e)
        return f"Error executing {name}: {str(e)}"

def _analyze_data(args: Dict[str, Any], state: Any) -> str:
    """Analyze an Excel or CSV file using pandas."""
    source_name = args.get("source")
    # Search for the file in the live folder
    from src.config import get_settings
    settings = get_settings()
    file_path = settings.live_folder / source_name
    
    if not file_path.exists():
        return f"File '{source_name}' not found in the Live Folder ({settings.live_folder})."

    # Load data
    ext = file_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        return f"Unsupported file format: {ext}"

    op = args.get("operation")
    if op == "summary":
        summary = df.describe(include='all').to_string()
        return f"Data Summary for {source_name}:\n```\n{summary}\n```\nColumns: {', '.join(df.columns)}"
    
    if op == "group_by":
        col = args.get("column")
        func = args.get("agg_func", "sum")
        if col not in df.columns:
            return f"Column '{col}' not found in {source_name}."
        
        # Identify numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=['number']).columns
        if numeric_cols.empty:
            return "No numeric columns found for aggregation."
            
        grouped = df.groupby(col)[numeric_cols].agg(func).reset_index()
        return f"Grouped result for {col} ({func}):\n```\n{grouped.head(10).to_string()}\n```"

    return f"Operation '{op}' not implemented yet."

def _create_visual(args: Dict[str, Any]) -> str:
    """Generate a chart and save as image."""
    chart_type = args.get("chart_type")
    title = args.get("title")
    labels = args.get("labels")
    values = args.get("values")

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    if chart_type == "bar":
        sns.barplot(x=labels, y=values, hue=labels, palette="viridis", legend=False)
    elif chart_type == "line":
        sns.lineplot(x=labels, y=values, marker="o", color="#e94560")
    elif chart_type == "pie":
        plt.pie(values, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    elif chart_type == "scatter":
        sns.scatterplot(x=labels, y=values, color="#0f3460")
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save logic
    from src.config import get_settings
    settings = get_settings()
    report_dir = settings.data_dir / "reports" / "charts"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"chart_{int(os.path.getmtime('.'))}_{labels[0][:5] if labels else 'data'}.png".replace(" ", "_")
    output_path = report_dir / filename
    plt.savefig(output_path)
    plt.close()

    # Return markdown for image
    return f"Successfully generated '{title}'.\n\n![{title}](file://{output_path.absolute()})"
