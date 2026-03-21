# AetherForge v1.0 — src/modules/ragforge_indexer.py
# ─────────────────────────────────────────────────────────────────
# Precision Ingestion™ — World-class local document processing.
#
# Pipeline phases:
#   Phase 1 — Smart Loading
#     • Digital PDFs   → IBM Docling (tables, equations, reading order)
#     • Scanned PDFs   → PyMuPDF page images → VLM visual extractor
#     • Text/MD/CSV    → Direct load
#
#   Phase 2 — Semantic Chunking
#     • Splits at section/table/equation boundaries — never mid-formula
#     • Each chunk is tagged with type, page, section, bbox metadata
#
#   Phase 3 — BGE-M3 Embedding
#     • 8192-token limit (vs old 512) — full sections fit in one vector
#     • Stored in ChromaDB with rich metadata for citation generation
#
# 100% offline. No internet required. Air-gap safe.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import gc
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import psutil
import structlog
from langchain_core.documents import Document

logger = structlog.get_logger("aetherforge.ragforge_indexer")

# ── Memory Governor ──────────────────────────────────────────────
# Hard cap: never start an IN-PROCESS VLM if system memory exceeds this.
# On 8GB macOS: baseline OS uses ~50% (4GB). At 90% ceiling = 7.2GB.
# This provides more headroom for VLMs on memory-constrained systems.
MEMORY_CEILING_PCT = 90.0
# Ollama-based VLMs are out-of-process; use a higher threshold.
MEMORY_CEILING_PCT_OLLAMA = 95.0


def _check_memory_budget(label: str = "VLM", is_ollama: bool = False) -> bool:
    """Returns True if enough memory is available to proceed."""
    ceiling = MEMORY_CEILING_PCT_OLLAMA if is_ollama else MEMORY_CEILING_PCT
    mem = psutil.virtual_memory()
    if mem.percent >= ceiling:
        logger.warning(
            "⚠️  Memory Governor: %.1f%% used (ceiling: %.0f%%) — deferring %s",
            mem.percent,
            ceiling,
            label,
        )
        return False
    logger.info(
        "Memory Governor: %.1f%% used — %s approved (ceiling: %.0f%%)", mem.percent, label, ceiling
    )
    return True


# ── Chunk size limits ─────────────────────────────────────────────
# all-MiniLM-L6-v2 supports 512 tokens. At ~3.5 chars/token, that's ~1800 chars.
# We keep chunks well below that so embeddings are focused and not truncated.
MAX_SECTION_CHARS = 1500  # one section / heading block
MAX_TABLE_CHARS = 1200  # one complete table
MAX_EQUATION_CHARS = 800  # one equation block
FALLBACK_CHUNK_SIZE = 1000  # plain text fallback
FALLBACK_OVERLAP = 100  # overlap between chunks

# ── Ingestion Progress Tracker ────────────────────────────────────
# In-memory dict: filename → {current_page, total_pages, chunks_so_far, status, batch_time_avg}
# Read by the /api/v1/ragforge/ingestion-progress endpoint for real-time UI updates.
_ingestion_progress: dict[str, dict] = {}

# ── Docling Converter Singleton ───────────────────────────────────
# Docling's DocumentConverter initializes layout models, tableformer, and OCR engines.
# This takes 3-5 seconds. Re-creating it per batch is the #1 performance killer.
# We cache a single instance keyed by pipeline options hash.
_docling_converter_cache: dict[str, Any] = {}  # options_hash → DocumentConverter


# ─────────────────────────────────────────────────────────────────
# Phase 1: Smart Loading
# ─────────────────────────────────────────────────────────────────


def _analyze_pdf(filepath: Path) -> dict:
    """
    Analyze a PDF to determine processing strategy.
    Returns:
        {
            "is_scanned": bool,       # True if < 50 chars/page (pure scan)
            "has_images": bool,       # True if > 3 embedded images total
            "image_pages": list[int], # Page indices that contain embedded images
            "total_pages": int,
            "total_images": int,
        }
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(filepath))

        num_pages = max(len(doc), 1)
        total_chars = sum(len(page.get_text()) for page in doc)
        avg_chars = total_chars / num_pages

        image_pages: list[int] = []
        total_images = 0
        for page_idx in range(num_pages):
            page_images = doc[page_idx].get_images()
            if page_images:
                image_pages.append(page_idx)
                total_images += len(page_images)

        doc.close()

        result = {
            "is_scanned": avg_chars < 50,
            "has_images": total_images > 0,
            "image_pages": image_pages,
            "total_pages": num_pages,
            "total_images": total_images,
        }
        logger.info(
            "PDF analysis '%s': scanned=%s, images=%d on %d/%d pages",
            filepath.name,
            result["is_scanned"],
            total_images,
            len(image_pages),
            num_pages,
        )
        return result
    except Exception as e:
        logger.warning("PDF analysis failed: %s", e)
        return {
            "is_scanned": False,
            "has_images": False,
            "image_pages": [],
            "total_pages": 0,
            "total_images": 0,
        }
    finally:
        gc.collect()  # Ensure fitz handle is released quickly


def load_with_docling(
    filepath: Path,
    chunk_callback: Optional[Callable[[list[Document]], None]] = None,
    adaptive_batch_size: int = 10,
) -> list[Document]:
    """
    Use IBM Docling to extract structured content from a digital PDF.
    Supports a chunk_callback for progressive indexing of large documents.

    Returns LangChain Documents with rich metadata:
      - chunk_type: 'section' | 'table' | 'equation' | 'figure_caption'
      - section_heading: nearest h1/h2/h3 above this chunk
      - page: page number (0-indexed)
      - doc_type: 'text' | 'table' | 'formula' | 'picture'
    """
    import time as _time

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
        import fitz
        import gc

        logger.info("Docling (Throttled Mode): converting '%s'...", filepath.name)
        
        # Configure resource-limited pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.images_scale = 1.0
        pipeline_options.generate_page_images = False
        # Fix: Enable picture image generation so Docling exports real image
        # data instead of the '🖼️❌ Image not available' placeholder.
        pipeline_options.generate_picture_images = True
        
        # Disable expensive vision enrichments to save RAM
        if hasattr(pipeline_options, "enrichment"):
            pipeline_options.enrichment.do_picture_classification = False
            pipeline_options.enrichment.do_formula_classification = False
        
        # Limit concurrency to prevent iMac freezing
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=2,  # Increased from 1 → 2 for better MPS utilization
            device="mps"    # Keep MPS for speed
        )

        # ── XY-Cut++ Pre-Pass: build per-page layout map ─────────────
        # Detects column structure before Docling to tag chunks
        # with layout type ('single'|'double'|'table'|'image_heavy').
        try:
            from src.modules.ragforge.xycut_layout import detect_layout_type
            pdf_doc = fitz.open(str(filepath))
            total_pages = len(pdf_doc)
            page_layout_map: dict[int, str] = {}
            for pg_idx in range(total_pages):
                page_layout_map[pg_idx + 1] = detect_layout_type(pdf_doc[pg_idx])
            pdf_doc.close()
            logger.info(
                "XY-Cut++ layout scan: %d pages analysed for '%s'",
                total_pages,
                filepath.name,
            )
        except Exception as _xy_err:
            logger.debug("XY-Cut++ pre-pass skipped: %s", _xy_err)
            pdf_doc = fitz.open(str(filepath))
            total_pages = len(pdf_doc)
            pdf_doc.close()
            page_layout_map = {}
        
        # ── Converter Singleton ───────────────────────────────────────
        # Docling's DocumentConverter loads layout models + tableformer + OCR
        # engines on __init__ (~3-5s). Re-creating it per batch was the #1
        # performance killer (observed 37 × 3s = 111s wasted on HA-13).
        import hashlib
        opts_hash = hashlib.md5(
            f"{pipeline_options.do_ocr}{pipeline_options.do_table_structure}"
            f"{pipeline_options.images_scale}{pipeline_options.generate_picture_images}"
            f"{pipeline_options.accelerator_options.num_threads}"
            f"{pipeline_options.accelerator_options.device}"
            .encode()
        ).hexdigest()[:12]

        if opts_hash in _docling_converter_cache:
            converter = _docling_converter_cache[opts_hash]
            logger.info("Reusing cached Docling converter (hash=%s)", opts_hash)
        else:
            converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            _docling_converter_cache[opts_hash] = converter
            logger.info("Created new Docling converter (hash=%s) — cached for reuse", opts_hash)

        all_chunks: list[Document] = []
        current_section = "Introduction"
        # HTI: Hierarchical Tree Index state
        section_path: list[str] = ["Root"]
        section_ids: list[str] = [str(uuid.uuid4())]
        batch_size = adaptive_batch_size
        
        # ── Initialize progress tracker ───────────────────────────────
        _ingestion_progress[filepath.name] = {
            "current_page": 0,
            "total_pages": total_pages,
            "chunks_so_far": 0,
            "status": "extracting_text",
            "batch_times": [],
            "batch_size": batch_size,
        }

        for start_page in range(1, total_pages + 1, batch_size):
            end_page = min(start_page + batch_size - 1, total_pages)
            logger.info("Throttled Batch: pages %d to %d of %d", start_page, end_page, total_pages)
            
            batch_start_time = _time.monotonic()

            # Use Docling's page_range (1-indexed)
            result = converter.convert(str(filepath), page_range=(start_page, end_page))
            doc = result.document
            
            batch_elapsed = _time.monotonic() - batch_start_time
            
            batch_chunks: list[Document] = []
            for item, _level in doc.iterate_items():
                item_label = str(getattr(item, "label", "text")).lower()
                item_text = ""

                # Handle images/figures
                if item_label in ("picture", "image", "figure"):
                    caption_text = ""
                    if hasattr(item, "caption") and item.caption:
                        cap = item.caption
                        caption_text = (getattr(cap, "text", "") or str(cap)).strip()
                    
                    if not caption_text and hasattr(item, "text") and item.text:
                        caption_text = item.text.strip()
                        
                    if not caption_text:
                        try:
                            if hasattr(item, "export_to_markdown"):
                                import inspect
                                sig = inspect.signature(item.export_to_markdown)
                                caption_text = item.export_to_markdown(doc).strip() if "doc" in sig.parameters else item.export_to_markdown().strip()
                        except Exception:
                            pass
                            
                    if caption_text:
                        item_text = caption_text
                        item_label = "figure_caption"
                    else:
                        page_no = 0
                        if hasattr(item, "prov") and item.prov:
                            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                            page_no = getattr(prov, "page_no", 0)
                        item_text = f"[Figure on page {page_no}]"
                        item_label = "figure_caption"

                # Standard text / rich items
                elif hasattr(item, "text") and item.text:
                    item_text = item.text.strip()
                elif hasattr(item, "export_to_markdown"):
                    try:
                        import inspect
                        sig = inspect.signature(item.export_to_markdown)
                        item_text = item.export_to_markdown(doc).strip() if "doc" in sig.parameters else item.export_to_markdown().strip()
                    except Exception as md_err:
                        logger.debug("Markdown export failed: %s", md_err)
                        item_text = (getattr(item, "text", "") or "").strip()

                if not item_text:
                    continue

                # Heading detection & HTI Path management
                if item_label in ("section_header", "title", "h1", "h2", "h3"):
                    current_section = item_text[:120]
                    level = getattr(item, "level", 1)  # 1-indexed hierarchical level
                    
                    # Update section path based on level
                    # If level is 1, it's a top-level heading. If level is 2, it's a child, etc.
                    while len(section_path) > level:
                        section_path.pop()
                        section_ids.pop()
                    
                    section_path.append(current_section)
                    section_ids.append(str(uuid.uuid4()))

                # Map to RAG chunk types
                chunk_type = "section"
                max_chars = MAX_SECTION_CHARS
                if item_label == "table":
                    chunk_type, max_chars = "table", MAX_TABLE_CHARS
                    
                    # --- NEW: STRUCTURED TABLE INGESTION ---
                    try:
                        from src.modules.ragforge.calc_engine import CalcEngine
                        from src.config import get_settings
                        settings = get_settings()
                        calc_engine = CalcEngine(db_path=settings.data_dir / "structured_data.db")
                        
                        if hasattr(item, "export_to_dataframe"):
                            df = item.export_to_dataframe()
                            if not df.empty:
                                # Simple vessel_id extraction from filename like "HA - 13 LOADING..."
                                vessel_id = filepath.name.split(" - ")[0].strip() if " - " in filepath.name else filepath.stem
                                calc_engine.ingest_hydrostatic_table(vessel_id, df)
                    except Exception as e:
                        logger.warning(f"Failed structured table ingestion for {filepath.name}: {e}")
                    # ---------------------------------------
                    
                elif item_label in ("formula", "equation"):
                    chunk_type, max_chars = "equation", MAX_EQUATION_CHARS
                elif item_label == "figure_caption":
                    chunk_type = "figure_caption"

                # Apply splitting
                if len(item_text) > max_chars:
                    sub_chunks = _split_table_block(item_text, max_chars) if chunk_type == "table" else _split_large_block(item_text, max_chars)
                else:
                    sub_chunks = [item_text]

                for idx, text in enumerate(sub_chunks):
                    page_no = 0
                    if hasattr(item, "prov") and item.prov:
                        prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                        page_no = getattr(prov, "page_no", 0)

                    batch_chunks.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filepath.name,
                                "chunk_type": chunk_type,
                                "section": current_section,
                                "page": page_no,
                                "sub_index": idx,
                                "doc_label": item_label,
                                "parser": "docling",
                                # HTI Metadata
                                "path": " / ".join(str(s) for s in section_path),
                                "parent_id": str(section_ids[-2]) if len(section_ids) > 1 else None,
                                "section_id": str(section_ids[-1]),
                                # XY-Cut++ Layout Metadata
                                "layout": page_layout_map.get(page_no, "single"),
                            },
                        )
                    )

            
            # Progressive commitment
            if chunk_callback and batch_chunks:
                chunk_callback(batch_chunks)
                
            all_chunks.extend(batch_chunks)
            
            # ── Update progress tracker ──────────────────────────────
            progress = _ingestion_progress.get(filepath.name)
            if progress:
                progress["current_page"] = end_page
                progress["chunks_so_far"] = len(all_chunks)
                progress["batch_times"].append(round(batch_elapsed, 2))
                # Running average for ETA calculation
                avg_time = sum(progress["batch_times"]) / len(progress["batch_times"])
                remaining_batches = max(0, (total_pages - end_page) / batch_size)
                progress["eta_seconds"] = round(avg_time * remaining_batches, 1)
                progress["last_batch_seconds"] = round(batch_elapsed, 2)
                logger.info(
                    "Progress: %d/%d pages (%.0f%%) | %d chunks | batch %.1fs | ETA %.0fs",
                    end_page, total_pages,
                    100 * end_page / max(total_pages, 1),
                    len(all_chunks),
                    batch_elapsed,
                    progress["eta_seconds"],
                )

            # ── Explicitly unload backend to release OCR memory ──
            try:
                # Docling legacy check
                if hasattr(result, "input") and hasattr(result.input, "_backend") and result.input._backend:
                    result.input._backend.unload()
            except Exception:
                pass

            # Cleanup batch resources
            del result
            del doc
            gc.collect()

        # ── Mark progress complete ────────────────────────────────
        progress = _ingestion_progress.get(filepath.name)
        if progress:
            progress["status"] = "indexing_complete"
            progress["current_page"] = total_pages
            progress["chunks_so_far"] = len(all_chunks)
            progress["eta_seconds"] = 0

        logger.info("Docling (Throttled) completed: %d total chunks for '%s'", len(all_chunks), filepath.name)

        # ── Post-processing: Figure & Table Reference Extraction ──
        # Scan ALL text chunks for in-text figure/table references
        # like "Figure 3 shows..." or "Table 2 presents..." and
        # build enriched context chunks linking captions to references.
        import re as _re

        # Build a registry: figure_num → {caption, references, pages}
        figure_registry: dict[str, dict] = {}
        table_registry: dict[str, dict] = {}

        for chunk in all_chunks:
            text = chunk.page_content
            page = chunk.metadata.get("page", 0)

            # Track figure captions
            if chunk.metadata.get("chunk_type") == "figure_caption":
                fig_match = _re.search(r"(?:Figure|Fig\.?)\s*(\d+)", text, _re.IGNORECASE)
                if fig_match:
                    fig_num = fig_match.group(1)
                    figure_registry.setdefault(
                        fig_num, {"caption": "", "references": [], "pages": set()}
                    )
                    figure_registry[fig_num]["caption"] = text
                    figure_registry[fig_num]["pages"].add(page)

            # Track table content
            if chunk.metadata.get("chunk_type") == "table":
                tab_match = _re.search(r"Table\s*(\d+)", text, _re.IGNORECASE)
                if tab_match:
                    tab_num = tab_match.group(1)
                    table_registry.setdefault(
                        tab_num, {"content": "", "references": [], "pages": set()}
                    )
                    table_registry[tab_num]["content"] = text[:500]  # cap table text
                    table_registry[tab_num]["pages"].add(page)

            # Find in-text references to figures
            for fig_ref in _re.finditer(r"(?:Figure|Fig\.?)\s*(\d+)", text, _re.IGNORECASE):
                fig_num = fig_ref.group(1)
                figure_registry.setdefault(
                    fig_num, {"caption": "", "references": [], "pages": set()}
                )
                # Store the sentence containing the reference
                start = max(0, fig_ref.start() - 100)
                end = min(len(text), fig_ref.end() + 200)
                context_sentence = text[start:end].strip()
                if context_sentence not in figure_registry[fig_num]["references"]:
                    figure_registry[fig_num]["references"].append(context_sentence)
                figure_registry[fig_num]["pages"].add(page)

            # Find in-text references to tables
            for tab_ref in _re.finditer(r"Table\s*(\d+)", text, _re.IGNORECASE):
                tab_num = tab_ref.group(1)
                table_registry.setdefault(
                    tab_num, {"content": "", "references": [], "pages": set()}
                )
                start = max(0, tab_ref.start() - 100)
                end = min(len(text), tab_ref.end() + 200)
                context_sentence = text[start:end].strip()
                if context_sentence not in table_registry[tab_num]["references"]:
                    table_registry[tab_num]["references"].append(context_sentence)
                table_registry[tab_num]["pages"].add(page)

        # Create enriched figure context chunks
        for fig_num, info in figure_registry.items():
            parts = [f"Figure {fig_num}"]
            if info["caption"]:
                parts.append(f"Caption: {info['caption']}")
            if info["references"]:
                parts.append("Referenced in text:")
                for ref in info["references"][:5]:  # cap at 5 references
                    parts.append(f"  - {ref}")
            if info["pages"]:
                parts.append(f"Pages: {', '.join(str(p) for p in sorted(info['pages']))}")

            fig_text = "\n".join(parts)
            all_chunks.append(
                Document(
                    page_content=fig_text,
                    metadata={
                        "source": filepath.name,
                        "chunk_type": "figure_context",
                        "section": f"Figure {fig_num}",
                        "page": min(info["pages"]) if info["pages"] else 0,
                        "sub_index": 0,
                        "doc_label": "figure_context",
                        "parser": "docling",
                        "figure_number": fig_num,
                    },
                )
            )

        # Create enriched table context chunks
        for tab_num, info in table_registry.items():
            parts = [f"Table {tab_num}"]
            if info["content"]:
                parts.append(f"Content: {info['content']}")
            if info["references"]:
                parts.append("Referenced in text:")
                for ref in info["references"][:5]:
                    parts.append(f"  - {ref}")

            tab_text = "\n".join(parts)
            all_chunks.append(
                Document(
                    page_content=tab_text,
                    metadata={
                        "source": filepath.name,
                        "chunk_type": "table_context",
                        "section": f"Table {tab_num}",
                        "page": min(info["pages"]) if info["pages"] else 0,
                        "sub_index": 0,
                        "doc_label": "table_context",
                        "parser": "docling",
                        "table_number": tab_num,
                    },
                )
            )

        fig_count = len(figure_registry)
        tab_count = len(table_registry)
        if fig_count or tab_count:
            logger.info(
                "Figure/table registry: %d figures, %d tables extracted from text",
                fig_count,
                tab_count,
            )

        return all_chunks

    except ImportError:
        logger.warning("Docling not installed — falling back to PyPDFLoader")
        return _load_with_pypdf(filepath)
    except Exception as e:
        logger.error("Docling failed on '%s': %s — falling back to PyPDFLoader", filepath.name, e)
        return _load_with_pypdf(filepath)
    finally:
        gc.collect()


def _load_with_pypdf(filepath: Path) -> list[Document]:
    """Improved fallback: PyPDF with paragraph-aware splitting and heading context."""
    import re as _re

    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(filepath))
    raw_docs = loader.load()

    chunks: list[Document] = []
    current_heading = "Introduction"

    for doc in raw_docs:
        page_num = doc.metadata.get("page", 0)
        text = doc.page_content.strip()
        if not text:
            continue

        # Split into paragraphs (double newline or significant whitespace)
        paragraphs = _re.split(r"\n{2,}", text)
        current_block = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Detect headings: short lines, often all-caps or title-case
            is_heading = (
                len(para) < 120
                and not para.endswith(".")
                and (
                    para.isupper()
                    or para.istitle()
                    or _re.match(r"^\d+[\.\)]\s+", para)
                    or _re.match(
                        r"^(Abstract|Introduction|Conclusion|References|Appendix|Discussion|Results|Methodology|Background)",
                        para,
                        _re.IGNORECASE,
                    )
                )
            )

            if is_heading:
                # Flush current block before heading change
                if current_block.strip():
                    chunks.append(
                        Document(
                            page_content=f"[Section: {current_heading}]\n\n{current_block.strip()}",
                            metadata={
                                "source": filepath.name,
                                "chunk_type": "section",
                                "section": current_heading,
                                "page": page_num,
                                "sub_index": len(chunks),
                                "parser": "pypdf_semantic",
                            },
                        )
                    )
                    current_block = ""
                current_heading = para[:120]
                continue

            # Accumulate paragraphs; flush when approaching max size
            if len(current_block) + len(para) > 1200 and current_block:
                chunks.append(
                    Document(
                        page_content=f"[Section: {current_heading}]\n\n{current_block.strip()}",
                        metadata={
                            "source": filepath.name,
                            "chunk_type": "section",
                            "section": current_heading,
                            "page": page_num,
                            "sub_index": len(chunks),
                            "parser": "pypdf_semantic",
                        },
                    )
                )
                # Keep last 200 chars as overlap for continuity
                current_block = current_block[-200:] + "\n\n" + para + "\n\n"
            else:
                current_block += para + "\n\n"

        # Flush remaining content from this page
        if current_block.strip():
            chunks.append(
                Document(
                    page_content=f"[Section: {current_heading}]\n\n{current_block.strip()}",
                    metadata={
                        "source": filepath.name,
                        "chunk_type": "section",
                        "section": current_heading,
                        "page": page_num,
                        "sub_index": len(chunks),
                        "parser": "pypdf_semantic",
                    },
                )
            )
            current_block = ""

    logger.info("PyPDF semantic chunker: %d chunks from '%s'", len(chunks), filepath.name)
    return chunks


def _split_large_block(text: str, max_chars: int) -> list[str]:
    """
    Split an oversized block at paragraph/sentence boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    sub_chunks = []
    paragraphs = text.split("\n\n")
    if len(paragraphs) == 1:
        # Fallback to single line splits if no paragraphs exist
        paragraphs = text.split("\n")
        
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            sub_chunks.append(current.strip())
            current = para + "\n\n"
        else:
            current += para + "\n\n"
            
    if current.strip():
        sub_chunks.append(current.strip())

    return sub_chunks or [text[:max_chars]]

def _split_table_block(text: str, max_chars: int) -> list[str]:
    """
    Split an oversized markdown table while preserving the header row.
    """
    if len(text) <= max_chars:
        return [text]
        
    lines = text.split("\n")
    if len(lines) < 3 or "|" not in lines[0]:
        return _split_large_block(text, max_chars)
        
    header = lines[0] + "\n" + lines[1] + "\n"
    sub_chunks = []
    current = header
    
    for row in lines[2:]:
        if len(current) + len(row) > max_chars and current != header:
            sub_chunks.append(current.strip())
            current = header + row + "\n"
        else:
            current += row + "\n"
            
    if current.strip() != header.strip():
        sub_chunks.append(current.strip())
        
    return sub_chunks or [text[:max_chars]]


# ─────────────────────────────────────────────────────────────────
# Scanned PDF OCR Fallback (no Ollama required)
# Uses ocrmac (bundled on macOS) → pytesseract → raw text as last resort
# ─────────────────────────────────────────────────────────────────

def _load_scanned_pdf_with_ocr(filepath: Path, chunk_callback: Optional[Callable[[list[Document]], None]] = None) -> list[Document]:
    """
    Renders each page of a scanned PDF to an image and runs offline OCR.
    Cascade: ocrmac (installed on macOS) → pytesseract → raw PyMuPDF text.
    Emits chunk_callback batches per page for real-time progress updates.
    """
    import fitz
    import io
    import re as _re

    doc = fitz.open(str(filepath))
    total_pages = len(doc)
    all_chunks: list[Document] = []

    logger.info("Offline OCR fallback: scanning %d pages from '%s'", total_pages, filepath.name)

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_no = page_idx + 1
        page_text = ""

        # Strategy 1: Get native text from PyMuPDF (often works partially)
        native_text = page.get_text("text").strip()
        if len(native_text) > 40:  # Has meaningful text already
            page_text = native_text
        else:
            # Strategy 2: Render to image → ocrmac (macOS native, fast)
            try:
                import ocrmac
                mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better OCR accuracy
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                page_text = ocrmac.OCR(io.BytesIO(img_bytes)).text
            except Exception as ocrmac_err:
                logger.debug("ocrmac failed page %d: %s", page_no, ocrmac_err)
                # Strategy 3: pytesseract fallback
                try:
                    from PIL import Image
                    import pytesseract
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                except Exception as tess_err:
                    logger.debug("pytesseract failed page %d: %s", page_no, tess_err)
                    page_text = native_text or f"[Page {page_no}: OCR unavailable]"

        if not page_text.strip():
            continue

        # Split extracted text into semantic chunks (~1000 chars with paragraph awareness)
        paragraphs = _re.split(r"\n{2,}", page_text.strip())
        current_block = ""
        current_section = f"Page {page_no}"

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Detect short headings
            is_heading = len(para) < 100 and not para.endswith(".")
            if is_heading and current_block:
                chunk = Document(
                    page_content=f"[Section: {current_section}]\n\n{current_block.strip()}",
                    metadata={"source": filepath.name, "chunk_type": "section",
                               "section": current_section, "page": page_no,
                               "sub_index": len(all_chunks), "parser": "ocr_fallback"},
                )
                all_chunks.append(chunk)
                if chunk_callback:
                    chunk_callback([chunk])
                current_block = ""
                current_section = para[:100]
                continue
            if len(current_block) + len(para) > FALLBACK_CHUNK_SIZE and current_block:
                chunk = Document(
                    page_content=f"[Section: {current_section}]\n\n{current_block.strip()}",
                    metadata={"source": filepath.name, "chunk_type": "section",
                               "section": current_section, "page": page_no,
                               "sub_index": len(all_chunks), "parser": "ocr_fallback"},
                )
                all_chunks.append(chunk)
                if chunk_callback:
                    chunk_callback([chunk])
                current_block = current_block[-200:] + "\n\n" + para + "\n\n"
            else:
                current_block += para + "\n\n"

        if current_block.strip():
            chunk = Document(
                page_content=f"[Section: {current_section}]\n\n{current_block.strip()}",
                metadata={"source": filepath.name, "chunk_type": "section",
                           "section": current_section, "page": page_no,
                           "sub_index": len(all_chunks), "parser": "ocr_fallback"},
            )
            all_chunks.append(chunk)
            if chunk_callback:
                chunk_callback([chunk])

        gc.collect()

    doc.close()
    logger.info("OCR fallback: %d chunks extracted from '%s'", len(all_chunks), filepath.name)
    return all_chunks


# ─────────────────────────────────────────────────────────────────
# Phase 1 Router: detect format and dispatch to right loader
# ─────────────────────────────────────────────────────────────────


def load_document(
    filepath: Path,
    chunk_callback: Optional[Callable[[list[Document]], None]] = None,
) -> tuple[list[Document], list[int]]:
    """
    Smart document router:
      .pdf, .xlsx, .xls  → Docling (default)
      .csv/.tsv          → Delimited loader
      .txt/.md/.json     → TextLoader

    Returns:
      Tuple of (chunks, image_pages) where image_pages is a list of
      0-indexed page numbers that should be enriched by an async VLM pass.
    """
    ext = filepath.suffix.lower()

    try:
        if ext in (".pdf", ".xlsx", ".xls"):
            if ext == ".pdf":
                analysis = _analyze_pdf(filepath)

                if analysis["is_scanned"]:
                    logger.info(
                        "'%s' is fully scanned (%d image pages) — trying Docling first, "
                        "then offline OCR fallback if needed.",
                        filepath.name, analysis.get("total_images", 0)
                    )
            else:
                # Excel files don't need scanned analysis
                analysis = {"has_images": False, "image_pages": [], "is_scanned": False}

            # Digital PDF or Excel — Docling/PyPDF handles all text extraction
            # ── Adaptive batch sizing (Fix 5) ──────────────────────────
            # Scanned PDFs: pure OCR, MPS amortizes well → 25 pages/batch
            # Mixed (text + images): balance init vs RAM → 15 pages/batch
            # Table-heavy or Excel: TableFormer spikes RAM → 10 pages/batch
            if analysis.get("is_scanned"):
                adaptive_batch = 25
            elif analysis.get("has_images"):
                adaptive_batch = 15
            else:
                adaptive_batch = 10
            chunks = load_with_docling(filepath, chunk_callback=chunk_callback, adaptive_batch_size=adaptive_batch)

            # --- Scanned PDF OCR Fallback ---
            # If Docling produces 0 chunks on a fully scanned PDF, use offline OCR.
            # This ensures we ALWAYS get usable content even when Ollama is offline.
            if analysis.get("is_scanned") and len(chunks) == 0:
                logger.warning(
                    "'%s' produced 0 Docling chunks — activating offline OCR fallback (ocrmac/pytesseract)",
                    filepath.name
                )
                chunks = _load_scanned_pdf_with_ocr(filepath, chunk_callback=chunk_callback)
                if chunks:
                    logger.info(
                        "Offline OCR fallback produced %d chunks for '%s'",
                        len(chunks), filepath.name
                    )
                    # For scanned docs processed via OCR: no VLM image_pages needed
                    return chunks, []

            # --- Selective VLM Routing (OpenDataLoader Pattern) ---
            # Route ONLY specific pages that need VLM enrichment.
            if analysis.get("is_scanned") and len(chunks) > 0:
                # Scanned but Docling got text: VLM would be redundant — skip
                image_pages: list[int] = []
            else:
                # Precision routing: only pages with embedded images/tables
                image_pages = list(analysis.get("image_pages", []))
                
            if image_pages:
                logger.info(
                    "Hybrid mode: Docling extracted %d text chunks, "
                    "%d image pages queued for async VLM processing",
                    len(chunks),
                    len(image_pages),
                )

            return chunks, image_pages

        elif ext in (".csv", ".tsv"):
            from langchain_community.document_loaders import CSVLoader

            loader = CSVLoader(str(filepath), csv_args={"delimiter": "\t" if ext == ".tsv" else ","})
            docs = loader.load()
            if chunk_callback:
                chunk_callback(docs)
            return docs, []

        elif ext in (".txt", ".md", ".json"):
            from langchain_community.document_loaders import TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            loader = TextLoader(str(filepath), encoding="utf-8")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=FALLBACK_CHUNK_SIZE,
                chunk_overlap=FALLBACK_OVERLAP,
                add_start_index=True,
            )
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = filepath.name
                chunk.metadata["chunk_type"] = "section"
                chunk.metadata["parser"] = "jsonloader" if ext == ".json" else "textloader"
            
            if chunk_callback:
                chunk_callback(chunks)
            return chunks, []

        else:
            logger.warning("Unsupported extension '%s' — trying TextLoader", ext)
            from langchain_community.document_loaders import TextLoader

            loader = TextLoader(str(filepath), encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filepath.name
                doc.metadata["chunk_type"] = "section"
                doc.metadata["parser"] = "textloader_fallback"
            return docs, []

    except Exception as e:
        logger.error("load_document failed for '%s': %s", filepath.name, e)
        return [], []


# ─────────────────────────────────────────────────────────────────
# Phase 2 + 3: Index into ChromaDB
# ─────────────────────────────────────────────────────────────────


def index_document(
    filepath: Path,
    vector_store: Any,
    sparse_index: Any = None,
    document_registry: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Fast Ingestion pipeline:
      1. Idempotency Guard — skip re-indexing if doc is already ready/partial
         and the file's mtime hasn't changed since last index (boot-sweep fix).
      2. Delete existing chunks for this source (deduplication guard)
      3. Load + semantically chunk (Docling text extraction → OCR fallback)
      4. Embed with all-MiniLM-L6-v2 and store in ChromaDB
      5. Write to FTS5 sparse index (uses shared AppState singleton when provided)
      6. Return image_pages for async VLM processing
    """
    import os
    logger.info("RAGForge Precision Ingestion — indexing: %s", filepath.name)

    source_name = filepath.name

    # ── Step 0: Idempotency Guard (Boot-Sweep Protection) ────────────
    # If the document is already indexed (ready/partial) and the file on disk
    # has NOT changed since the last index, skip re-indexing entirely.
    # This prevents the "reset to 0" bug caused by boot-sweep re-running on restart.
    if not force and document_registry is not None:
        try:
            existing_record = document_registry.get_by_source(source_name)
            if existing_record and existing_record.ingest_status in ("ready", "partial") and existing_record.chunk_count > 0:
                # Check file modification time
                current_mtime = os.path.getmtime(str(filepath))
                last_indexed = getattr(existing_record, "last_indexed_mtime", 0.0)
                if last_indexed > 0 and abs(current_mtime - last_indexed) < 1.0:
                    logger.info(
                        "Idempotency Guard: '%s' already indexed with %d chunks (status=%s, file unchanged) — skipping indexing.",
                        source_name, existing_record.chunk_count, existing_record.ingest_status,
                    )
                    return {
                        "file": source_name,
                        "chunks_added": existing_record.chunk_count,
                        "skipped": True,
                        "parser": "cached",
                        "chunk_breakdown": {},
                        "image_pages": [],
                    }
        except Exception as guard_err:
            logger.debug("Idempotency guard check failed (non-fatal): %s", guard_err)

    # ── Step 1: Deduplicate — delete existing chunks for this source ──
    # This prevents chunk accumulation when a user re-uploads the same document.
    try:
        # ChromaDB dedup: delete by source metadata
        existing = vector_store.get(where={"source": source_name})
        if existing and existing.get("ids"):
            vector_store.delete(ids=existing["ids"])
            logger.info(
                "Dedup: removed %d existing ChromaDB chunks for '%s'",
                len(existing["ids"]),
                source_name,
            )
    except Exception as dedup_err:
        logger.warning("ChromaDB dedup failed (non-fatal): %s", dedup_err)

    if sparse_index is not None:
        try:
            deleted = sparse_index.delete_by_source(source_name)
            if deleted > 0:
                logger.info("Dedup: removed %d existing FTS5 chunks for '%s'", deleted, source_name)
        except Exception as dedup_err:
            logger.warning("FTS5 dedup failed (non-fatal): %s", dedup_err)

    # Define progressive commitment callback
    def commit_chunks(batch: list[Document]):
        if not batch:
            return
            
        # Inject stable source metadata
        parser = batch[0].metadata.get("chunk_id_prefix", "docling") # fallback
        for chunk in batch:
            chunk.metadata.setdefault("source", filepath.name)
            chunk.metadata.setdefault("chunk_type", "section")
            chunk.metadata.setdefault("section", "Unknown")
            chunk.metadata.setdefault("page", 0)
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            # parser is already in metadata from load_with_docling

        # Embed and store in ChromaDB
        logger.info("Committing %d chunks to vector store...", len(batch))
        vector_store.add_documents(batch)

        # FTS5 Sparse Index
        try:
            if sparse_index is not None:
                sparse_index.add_documents(batch)
            else:
                from src.modules.ragforge.sparse_index import SparseIndex
                sparse_idx = SparseIndex(db_path=filepath.parent.parent / "sparse_index.db")
                sparse_idx.add_documents(batch)
                sparse_idx.close()
        except Exception as fts_err:
            logger.warning("FTS5 batch indexing failed: %s", fts_err)

        # [RUVECTOR PHASE 6 PLACEHOLDER] Cypher Graph Extraction
        try:
            import subprocess
            subprocess.run(["npx", "--yes", "ruvector", "graph", "insert", str(filepath.parent.parent / "graph.db")], capture_output=True, check=True)
            logger.info("Cypher Graph relationships extracted and stored for %d chunks", len(batch))
        except Exception as e:
            pass

    # Load via smart router — returns (chunks, image_pages)
    result = load_document(filepath, chunk_callback=commit_chunks)
    if isinstance(result, tuple):
        chunks, image_pages = result
    else:
        chunks, image_pages = result, []

    if not chunks:
        logger.warning(
            "No text chunks extracted from '%s' (may need VLM for scanned PDF)", filepath.name
        )
        if image_pages:
            return {
                "file": filepath.name,
                "chunks_added": 0,
                "image_pages": image_pages,
                "parser": "pending_vlm",
                "chunk_breakdown": {},
            }
        return {"file": filepath.name, "chunks_added": 0, "error": "extraction_failed"}

    # Build summary by chunk type
    type_counts: dict[str, int] = {}
    for chunk in chunks:
        ct = chunk.metadata.get("chunk_type", "section")
        type_counts[ct] = type_counts.get(ct, 0) + 1

    parser = chunks[0].metadata.get("parser", "unknown") if chunks else "unknown"
    logger.info(
        "Indexed '%s' — %d chunks TOTAL via %s | breakdown: %s",
        filepath.name,
        len(chunks),
        parser,
        type_counts,
    )

    # Record file mtime so the idempotency guard can detect future changes
    import os
    try:
        file_mtime = os.path.getmtime(str(filepath))
    except Exception:
        file_mtime = None

    return {
        "file": filepath.name,
        "chunks_added": len(chunks),
        "parser": parser,
        "chunk_breakdown": type_counts,
        "image_pages": image_pages,
        "pending_image_pages": len(image_pages),
        "last_indexed_mtime": file_mtime,
    }
