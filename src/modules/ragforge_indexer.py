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

import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger("aetherforge.ragforge_indexer")

# ── Chunk size limits ─────────────────────────────────────────────
# BGE-M3 supports 8192 tokens. At ~3.5 chars/token, that's ~28 000 chars.
# We keep chunks well below that so embeddings are focused, not diluted.
MAX_SECTION_CHARS = 6000   # one section / heading block
MAX_TABLE_CHARS = 4000     # one complete table
MAX_EQUATION_CHARS = 2000  # one equation block
FALLBACK_CHUNK_SIZE = 1500  # plain text fallback (better than old 1000)
FALLBACK_OVERLAP = 150      # reduced from 200; less semantic confusion


# ─────────────────────────────────────────────────────────────────
# Phase 1: Smart Loading
# ─────────────────────────────────────────────────────────────────

def _should_use_vlm(filepath: Path) -> bool:
    """
    Heuristic to determine if we need the Visual Language Model:
    1. Scanned PDF: < 50 chars/page on average
    2. Image-heavy digital PDF: > 5 embedded images total (meaning it has charts/figures)
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(filepath))
        
        num_pages = max(len(doc), 1)
        total_chars = sum(len(page.get_text()) for page in doc)
        total_images = sum(len(page.get_images()) for page in doc)
        
        avg_chars = total_chars / num_pages
        doc.close()
        
        # Route to VLM if it's literally a scan, or if it has embedded figures to read
        if avg_chars < 50 or total_images > 5:
            logger.info("Routing '%s' to VLM (avg_chars: %.0f, total_images: %d)", filepath.name, avg_chars, total_images)
            return True
        return False
    except Exception as e:
        logger.warning("VLM heuristic failed: %s", e)
        return False  # assume digital text-only if we can't tell


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

            # Skip picture/image items — they have no text content to embed.
            # PictureItem.export_to_markdown() requires `doc` arg and returns
            # only a placeholder; VLM handles these via the scanned PDF path.
            if item_label in ("picture", "image", "figure"):
                continue

            # Extract text content based on item type
            if hasattr(item, "text") and item.text:
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

                chunks.append(Document(
                    page_content=text,
                    metadata={
                        "source": filepath.name,
                        "chunk_type": chunk_type,
                        "section": current_section,
                        "page": page_num,
                        "sub_index": i,
                        "doc_label": item_label,
                        "parser": "docling",
                    }
                ))

        logger.info("Docling extracted %d semantic chunks from '%s'", len(chunks), filepath.name)
        return chunks

    except ImportError:
        logger.warning("Docling not installed — falling back to PyPDFLoader")
        return _load_with_pypdf(filepath)
    except Exception as e:
        logger.error("Docling failed on '%s': %s — falling back to PyPDFLoader", filepath.name, e)
        return _load_with_pypdf(filepath)


def load_scanned_with_vlm(filepath: Path) -> list[Document]:
    """
    For scanned / image-heavy PDFs: render each page with PyMuPDF,
    send to VLM (Qwen2-VL via Ollama) for visual text extraction.
    Falls back to PyPDFLoader if Ollama is not running.
    """
    try:
        import fitz  # PyMuPDF
        from src.modules.ragforge.vlm_processor import AcademicVLMProcessor
        import asyncio

        logger.info("VLM path: rendering '%s' pages as images...", filepath.name)
        pdf_doc = fitz.open(str(filepath))
        vlm = AcademicVLMProcessor()
        chunks: list[Document] = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            # Render at 2x resolution for better OCR quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            # Run async VLM in sync context
            try:
                loop = asyncio.new_event_loop()
                extracted = loop.run_until_complete(vlm.extract_academic_content(img_bytes))
                loop.close()
            except Exception as vlm_err:
                logger.warning("VLM failed on page %d: %s", page_num, vlm_err)
                extracted = {"text": "", "tables": [], "equations": []}

            # Main page text
            if extracted.get("text"):
                chunks.append(Document(
                    page_content=extracted["text"],
                    metadata={
                        "source": filepath.name,
                        "chunk_type": "section",
                        "section": extracted.get("section_heading", f"Page {page_num + 1}"),
                        "page": page_num,
                        "parser": "vlm",
                    }
                ))

            # Tables extracted by VLM
            for table_text in extracted.get("tables", []):
                if table_text.strip():
                    chunks.append(Document(
                        page_content=table_text,
                        metadata={
                            "source": filepath.name,
                            "chunk_type": "table",
                            "section": extracted.get("section_heading", f"Page {page_num + 1}"),
                            "page": page_num,
                            "parser": "vlm",
                        }
                    ))

            # Equations extracted by VLM
            for eq_text in extracted.get("equations", []):
                if eq_text.strip():
                    chunks.append(Document(
                        page_content=eq_text,
                        metadata={
                            "source": filepath.name,
                            "chunk_type": "equation",
                            "section": extracted.get("section_heading", f"Page {page_num + 1}"),
                            "page": page_num,
                            "parser": "vlm",
                        }
                    ))

        pdf_doc.close()
        logger.info("VLM extracted %d chunks from '%s'", len(chunks), filepath.name)
        return chunks

    except ImportError as ie:
        logger.warning("VLM/PyMuPDF not available (%s) — falling back to PyPDFLoader", ie)
        return _load_with_pypdf(filepath)
    except Exception as e:
        logger.error("VLM processing failed: %s — falling back to PyPDFLoader", e)
        return _load_with_pypdf(filepath)


def _load_with_pypdf(filepath: Path) -> list[Document]:
    """Legacy fallback: PyPDFLoader with improved chunking."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(str(filepath))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["parser"] = "pypdf_fallback"
        chunk.metadata["chunk_type"] = "section"
        chunk.metadata.setdefault("source", filepath.name)
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
      .pdf  → Docling (default) or VLM (scanned fallback)
      .csv  → CSVLoader
      .txt/.md → TextLoader
    """
    ext = filepath.suffix.lower()

    try:
        if ext == ".pdf":
            # Decide: digital text-only or VLM-required (scanned / image-heavy)?
            if _should_use_vlm(filepath):
                logger.info("'%s' routed to VLM (scanned/image-heavy)", filepath.name)
                return load_scanned_with_vlm(filepath)
            else:
                return load_with_docling(filepath)

        elif ext == ".csv":
            from langchain_community.document_loaders import CSVLoader
            loader = CSVLoader(str(filepath))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filepath.name
                doc.metadata["chunk_type"] = "table"
                doc.metadata["parser"] = "csvloader"
            return docs

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
            return chunks

        else:
            logger.warning("Unsupported extension '%s' — trying TextLoader", ext)
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(str(filepath), encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filepath.name
                doc.metadata["chunk_type"] = "section"
                doc.metadata["parser"] = "textloader_fallback"
            return docs

    except Exception as e:
        logger.error("load_document failed for '%s': %s", filepath.name, e)
        return []


# ─────────────────────────────────────────────────────────────────
# Phase 2 + 3: Index into ChromaDB
# ─────────────────────────────────────────────────────────────────

def index_document(filepath: Path, vector_store: Any) -> dict[str, Any]:
    """
    Full Precision Ingestion™ pipeline:
      1. Load + semantically chunk (Phase 1+2)
      2. Embed with BGE-M3 and store in ChromaDB (Phase 3)

    Returns indexing summary with chunk breakdown by type.
    """
    logger.info("RAGForge Precision Ingestion — indexing: %s", filepath.name)

    # Load via smart router
    chunks = load_document(filepath)
    if not chunks:
        logger.error("No content extracted from '%s'", filepath.name)
        return {"file": filepath.name, "chunks_added": 0, "error": "extraction_failed"}

    # Inject stable source metadata
    for chunk in chunks:
        chunk.metadata.setdefault("source", filepath.name)
        chunk.metadata.setdefault("chunk_type", "section")
        chunk.metadata.setdefault("section", "Unknown")
        chunk.metadata.setdefault("page", 0)
        # Add stable unique doc_id per chunk for dedup
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    # Embed and store in ChromaDB (BGE-M3 used via vector_store.embedding_function)
    logger.info("Embedding %d chunks with BGE-M3...", len(chunks))
    vector_store.add_documents(chunks)

    # Build summary by chunk type
    type_counts: dict[str, int] = {}
    for chunk in chunks:
        ct = chunk.metadata.get("chunk_type", "section")
        type_counts[ct] = type_counts.get(ct, 0) + 1

    parser = chunks[0].metadata.get("parser", "unknown") if chunks else "unknown"
    logger.info(
        "Indexed '%s' — %d chunks via %s | breakdown: %s",
        filepath.name, len(chunks), parser, type_counts
    )

    return {
        "file": filepath.name,
        "chunks_added": len(chunks),
        "parser": parser,
        "chunk_breakdown": type_counts,
    }
