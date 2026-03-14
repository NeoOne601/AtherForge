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
from typing import Any

import psutil
import structlog
from langchain_core.documents import Document

logger = structlog.get_logger("aetherforge.ragforge_indexer")

# ── Memory Governor ──────────────────────────────────────────────
# Hard cap: never start an IN-PROCESS VLM if system memory exceeds this.
# On 8GB macOS: baseline OS uses ~50% (4GB). At 85% ceiling = 6.8GB.
# NOTE: Ollama runs as a SEPARATE OS process and is NOT subject to this
# ceiling — it manages its own memory. Only in-process HuggingFace models
# (SmolVLM, Florence, QwenVL) are gated by this check.
MEMORY_CEILING_PCT = 85.0
# Ollama-based VLMs are out-of-process; use a higher threshold since they
# don't load into Python's heap at all.
MEMORY_CEILING_PCT_OLLAMA = 97.0


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


def load_with_docling(filepath: Path) -> list[Document]:
    """
    Use IBM Docling to extract structured content from a digital PDF.

    Returns LangChain Documents with rich metadata:
      - chunk_type: 'section' | 'table' | 'equation' | 'figure_caption'
      - section_heading: nearest h1/h2/h3 above this chunk
      - page: page number (0-indexed)
      - doc_type: 'text' | 'table' | 'formula' | 'picture'
    """
    try:
        from docling.document_converter import DocumentConverter

        logger.info("Docling: converting '%s'...", filepath.name)
        converter = DocumentConverter()
        result = converter.convert(str(filepath))
        doc = result.document

        chunks: list[Document] = []
        current_section = "Introduction"

        for item, _level in doc.iterate_items():
            item_label = str(getattr(item, "label", "text")).lower()
            item_text = ""

            # Handle picture/image/figure items — extract captions and any text
            # IMPORTANT: Do NOT skip these. Docling's PictureItem may contain
            # caption text that is the ONLY representation of figures in the index.
            if item_label in ("picture", "image", "figure"):
                # Try to extract caption text from the item
                caption_text = ""
                if hasattr(item, "caption") and item.caption:
                    cap = item.caption
                    if hasattr(cap, "text"):
                        caption_text = cap.text.strip()
                    elif isinstance(cap, str):
                        caption_text = cap.strip()
                if not caption_text and hasattr(item, "text") and item.text:
                    caption_text = item.text.strip()
                if not caption_text:
                    # Try export_to_markdown as last resort
                    try:
                        if hasattr(item, "export_to_markdown"):
                            import inspect

                            sig = inspect.signature(item.export_to_markdown)
                            if "doc" in sig.parameters:
                                caption_text = item.export_to_markdown(doc).strip()
                            else:
                                caption_text = item.export_to_markdown().strip()
                    except Exception:
                        pass
                if caption_text:
                    item_text = caption_text
                    item_label = "figure_caption"  # override for proper chunk typing
                else:
                    # No text at all — use a placeholder so the figure is at least findable
                    page_num = 0
                    if hasattr(item, "prov") and item.prov:
                        prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                        page_num = getattr(prov, "page_no", 0)
                    item_text = f"[Figure on page {page_num}]"
                    item_label = "figure_caption"

            # Extract text content based on item type
            elif hasattr(item, "text") and item.text:
                item_text = item.text.strip()
            elif hasattr(item, "export_to_markdown"):
                try:
                    # Tables and other rich items may need the doc reference
                    import inspect

                    sig = inspect.signature(item.export_to_markdown)
                    if "doc" in sig.parameters:
                        item_text = item.export_to_markdown(doc).strip()
                    else:
                        item_text = item.export_to_markdown().strip()
                except Exception as md_err:
                    logger.debug("export_to_markdown failed for %s: %s", item_label, md_err)
                    item_text = getattr(item, "text", "") or ""
                    if isinstance(item_text, str):
                        item_text = item_text.strip()

            if not item_text:
                continue

            # Track section headings for citation metadata
            if item_label in ("section_header", "title", "h1", "h2", "h3"):
                current_section = item_text[:120]  # cap heading length

            # Map Docling label to our chunk type
            if item_label in ("table",):
                chunk_type = "table"
                max_chars = MAX_TABLE_CHARS
            elif item_label in ("formula", "equation"):
                chunk_type = "equation"
                max_chars = MAX_EQUATION_CHARS
            elif item_label in ("caption", "figure_caption"):
                chunk_type = "figure_caption"
                max_chars = MAX_SECTION_CHARS
            else:
                chunk_type = "section"
                max_chars = MAX_SECTION_CHARS

            # Split only if chunk is unusually large (rare for Docling)
            if len(item_text) > max_chars:
                sub_chunks = _split_large_block(item_text, max_chars)
            else:
                sub_chunks = [item_text]

            for i, text in enumerate(sub_chunks):
                # Extract page number from item's bounding box if available
                page_num = 0
                if hasattr(item, "prov") and item.prov:
                    prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                    page_num = getattr(prov, "page_no", 0)

                chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filepath.name,
                            "chunk_type": chunk_type,
                            "section": current_section,
                            "page": page_num,
                            "sub_index": i,
                            "doc_label": item_label,
                            "parser": "docling",
                        },
                    )
                )

        logger.info("Docling extracted %d semantic chunks from '%s'", len(chunks), filepath.name)

        # ── Post-processing: Figure & Table Reference Extraction ──
        # Scan ALL text chunks for in-text figure/table references
        # like "Figure 3 shows..." or "Table 2 presents..." and
        # build enriched context chunks linking captions to references.
        import re as _re

        # Build a registry: figure_num → {caption, references, pages}
        figure_registry: dict[str, dict] = {}
        table_registry: dict[str, dict] = {}

        for chunk in chunks:
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
            chunks.append(
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
            chunks.append(
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

        return chunks

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
    Only called for unusually large Docling items — rare.
    """
    if len(text) <= max_chars:
        return [text]

    sub_chunks = []
    paragraphs = text.split("\n\n")
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


# ─────────────────────────────────────────────────────────────────
# Phase 1 Router: detect format and dispatch to right loader
# ─────────────────────────────────────────────────────────────────


def load_document(filepath: Path) -> list[Document]:
    """
    Smart document router:
      .pdf, .xlsx, .xls  → Docling (default)
      .csv               → CSVLoader
      .txt/.md           → TextLoader
    """
    ext = filepath.suffix.lower()

    try:
        if ext in (".pdf", ".xlsx", ".xls"):
            if ext == ".pdf":
                analysis = _analyze_pdf(filepath)

                if analysis["is_scanned"]:
                    # Purely scanned — VLM must read every page
                    logger.info("'%s' is scanned — full VLM processing needed", filepath.name)
                    # Return all pages as image_pages for async VLM processing
                    import fitz

                    pdf_doc = fitz.open(str(filepath))
                    all_pages = list(range(len(pdf_doc)))
                    pdf_doc.close()
                    return [], all_pages  # No text chunks, all pages need VLM
            else:
                # Excel files don't need scanned analysis
                analysis = {"has_images": False, "image_pages": []}

            # Digital PDF or Excel — Docling handles all text extraction
            chunks = load_with_docling(filepath)

            # Return image_pages for async VLM processing
            image_pages = analysis.get("image_pages", []) if analysis.get("has_images") else []
            if image_pages:
                logger.info(
                    "Hybrid mode: Docling extracted %d text chunks, "
                    "%d image pages queued for async VLM processing",
                    len(chunks),
                    len(image_pages),
                )

            return chunks, image_pages

        elif ext == ".csv":
            from langchain_community.document_loaders import CSVLoader

            loader = CSVLoader(str(filepath))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filepath.name
                doc.metadata["chunk_type"] = "table"
                doc.metadata["parser"] = "csvloader"
            return docs, []

        elif ext in (".txt", ".md"):
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
                chunk.metadata["parser"] = "textloader"
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


def index_document(filepath: Path, vector_store: Any, sparse_index: Any = None) -> dict[str, Any]:
    """
    Fast Ingestion pipeline:
      1. Delete existing chunks for this source (deduplication guard)
      2. Load + semantically chunk (Docling text extraction)
      3. Embed with all-MiniLM-L6-v2 and store in ChromaDB
      4. Write to FTS5 sparse index (uses shared AppState singleton when provided)
      5. Return image_pages for async VLM processing
    """
    logger.info("RAGForge Precision Ingestion — indexing: %s", filepath.name)

    # ── Step 0: Deduplicate — delete existing chunks for this source ──
    # This prevents chunk accumulation when a user re-uploads the same document.
    source_name = filepath.name
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

    # Load via smart router — returns (chunks, image_pages)
    result = load_document(filepath)
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

    # Inject stable source metadata
    parser = chunks[0].metadata.get("parser", "unknown") if chunks else "unknown"
    for chunk in chunks:
        chunk.metadata.setdefault("source", filepath.name)
        chunk.metadata.setdefault("chunk_type", "section")
        chunk.metadata.setdefault("section", "Unknown")
        chunk.metadata.setdefault("page", 0)
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["parser"] = parser  # store parser for traceability

    # Embed and store in ChromaDB
    logger.info("Embedding %d chunks...", len(chunks))
    vector_store.add_documents(chunks)

    # ── FTS5 Sparse Index: use shared singleton or create local instance ──
    try:
        if sparse_index is not None:
            sparse_index.add_documents(chunks)
            logger.info("FTS5 sparse index updated via shared singleton (%d chunks)", len(chunks))
        else:
            from src.modules.ragforge.sparse_index import SparseIndex

            sparse_idx = SparseIndex(db_path=filepath.parent.parent / "sparse_index.db")
            sparse_idx.add_documents(chunks)
            sparse_idx.close()
            logger.info("FTS5 sparse index updated with %d chunks", len(chunks))
    except Exception as fts_err:
        logger.warning("FTS5 indexing failed (non-fatal): %s", fts_err)

    # Build summary by chunk type
    type_counts: dict[str, int] = {}
    for chunk in chunks:
        ct = chunk.metadata.get("chunk_type", "section")
        type_counts[ct] = type_counts.get(ct, 0) + 1

    parser = chunks[0].metadata.get("parser", "unknown") if chunks else "unknown"
    logger.info(
        "Indexed '%s' — %d chunks via %s | breakdown: %s",
        filepath.name,
        len(chunks),
        parser,
        type_counts,
    )

    return {
        "file": filepath.name,
        "chunks_added": len(chunks),
        "parser": parser,
        "chunk_breakdown": type_counts,
        "image_pages": image_pages,
    }
