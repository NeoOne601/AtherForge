# AetherForge v1.0 — src/modules/ragforge/xycut_layout.py
# ─────────────────────────────────────────────────────────────────
# XY-Cut++ Multi-Column Layout Engine
# ─────────────────────────────────────────────────────────────────
#
# Implements the XY-Cut algorithm enhanced with gap-scoring to handle
# 2-column scientific paper layouts and complex magazine-style grids.
#
# Pipeline:
#   1. Get text blocks from fitz page.get_text("dict")
#   2. Project bounding boxes onto X-axis → find column split gaps
#   3. For each column: project onto Y-axis → sort in reading order
#   4. Merge text column-first, then row-descending
#
# Key advantage over vanilla Docling:
#   Docling merges columns by vertical interleaving → garbled text.
#   XY-Cut++ detects the column boundary and sorts each column
#   independently, preserving full human reading order.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger("aetherforge.xycut_layout")


@dataclass
class TextBlock:
    """A positioned text block from a PDF page."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_type: str = "text"     # "text" | "table" | "image"
    column: int = 0              # Assigned column index (0 = left)


@dataclass
class ColumnBlock:
    """A vertical column slice of the page."""
    x_start: float
    x_end: float
    blocks: list[TextBlock] = field(default_factory=list)


def _extract_raw_blocks(page: Any) -> list[TextBlock]:
    """
    Extract positioned text blocks from a PyMuPDF page.
    Filters out empty blocks and image-only blocks.
    """
    raw = page.get_text("dict")
    blocks: list[TextBlock] = []

    for blk in raw.get("blocks", []):
        blk_type = blk.get("type", 0)
        if blk_type == 1:  # Image block
            x0, y0, x1, y1 = blk.get("bbox", (0, 0, 0, 0))
            blocks.append(
                TextBlock(text="[IMAGE]", x0=x0, y0=y0, x1=x1, y1=y1, block_type="image")
            )
            continue

        # Text block: extract all lines
        lines = blk.get("lines", [])
        text_parts = []
        for line in lines:
            for span in line.get("spans", []):
                text_parts.append(span.get("text", "").strip())
        text = " ".join(text_parts).strip()

        if not text:
            continue

        x0, y0, x1, y1 = blk.get("bbox", (0, 0, 0, 0))
        blocks.append(TextBlock(text=text, x0=x0, y0=y0, x1=x1, y1=y1))

    return blocks


def _find_column_splits(
    blocks: list[TextBlock],
    page_width: float,
    min_gap_width: float = 30.0,
) -> list[float]:
    """
    Project all blocks onto X-axis and find significant empty gaps
    that represent column dividers.

    Returns sorted list of X-coordinates where columns split.
    """
    if not blocks:
        return []

    # Build occupancy array (5px resolution)
    resolution = 5
    width_slots = int(page_width / resolution) + 1
    occupancy = [0] * width_slots

    for blk in blocks:
        start = max(0, int(blk.x0 / resolution))
        end = min(width_slots - 1, int(blk.x1 / resolution))
        for i in range(start, end + 1):
            occupancy[i] += 1

    # Find continuous empty zones
    splits = []
    in_gap = False
    gap_start = 0

    for i, val in enumerate(occupancy):
        if val == 0 and not in_gap:
            in_gap = True
            gap_start = i
        elif val > 0 and in_gap:
            gap_end = i
            gap_width = (gap_end - gap_start) * resolution
            if gap_width >= min_gap_width:
                # Midpoint of the gap is the split line
                mid_x = ((gap_start + gap_end) / 2) * resolution
                splits.append(mid_x)
            in_gap = False

    return sorted(splits)


def _assign_columns(
    blocks: list[TextBlock],
    splits: list[float],
) -> list[ColumnBlock]:
    """
    Assign each text block to a column based on split X-coordinates.
    Returns columns sorted left-to-right.
    """
    # Build column boundaries
    boundaries: list[tuple[float, float]] = []
    prev = 0.0
    for split in splits:
        boundaries.append((prev, split))
        prev = split
    boundaries.append((prev, float("inf")))

    columns = [ColumnBlock(x_start=b[0], x_end=b[1]) for b in boundaries]

    for blk in blocks:
        # Assign to column whose center overlaps the block center
        blk_center = (blk.x0 + blk.x1) / 2
        for col_idx, (x_start, x_end) in enumerate(boundaries):
            if x_start <= blk_center < x_end:
                blk.column = col_idx
                columns[col_idx].blocks.append(blk)
                break

    return columns


def extract_page_text(page: Any, min_gap_width: float = 30.0) -> str:
    """
    High-level entry: extract text from a fitz page in correct reading order.

    Uses XY-Cut++ to handle multi-column layouts without column bleed.

    Args:
        page:          A PyMuPDF (fitz) page object.
        min_gap_width: Minimum pixel gap to detect a column boundary.

    Returns:
        Plain text string in correct human reading order.
    """
    try:
        blocks = _extract_raw_blocks(page)
        if not blocks:
            return ""

        page_width = page.rect.width
        splits = _find_column_splits(blocks, page_width, min_gap_width)

        if not splits:
            # Single column — sort top-to-bottom only
            sorted_blocks = sorted(blocks, key=lambda b: (b.y0, b.x0))
            return "\n".join(b.text for b in sorted_blocks if b.text and b.text != "[IMAGE]")

        columns = _assign_columns(blocks, splits)

        # Sort each column top-to-bottom; concatenate columns left-to-right
        text_parts: list[str] = []
        for col in columns:
            sorted_col = sorted(col.blocks, key=lambda b: b.y0)
            col_text = "\n".join(
                b.text for b in sorted_col if b.text and b.text != "[IMAGE]"
            )
            if col_text.strip():
                text_parts.append(col_text)

        result = "\n\n".join(text_parts)
        logger.debug(
            "XY-Cut++: %d blocks, %d columns, %d chars",
            len(blocks),
            len(columns),
            len(result),
        )
        return result

    except Exception as e:
        logger.warning("XY-Cut++ failed: %s — returning empty string", e)
        return ""


def detect_layout_type(page: Any) -> str:
    """
    Classify page layout for metadata tagging.

    Returns:
        "single" | "double" | "table" | "image_heavy"
    """
    try:
        blocks = _extract_raw_blocks(page)
        if not blocks:
            return "image_heavy"

        image_blocks = [b for b in blocks if b.block_type == "image"]
        text_blocks = [b for b in blocks if b.block_type == "text"]

        if len(image_blocks) > len(text_blocks):
            return "image_heavy"

        # Check for table-like content (many short lines with |)
        table_like = sum(1 for b in text_blocks if "|" in b.text or b.text.count("  ") > 3)
        if table_like > len(text_blocks) * 0.4:
            return "table"

        page_width = page.rect.width
        splits = _find_column_splits(text_blocks, page_width)
        if len(splits) >= 1:
            return "double"

        return "single"

    except Exception:
        return "single"
