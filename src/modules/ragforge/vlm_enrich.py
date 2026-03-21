from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import psutil
import structlog

logger = structlog.get_logger("aetherforge.vlm_enrich")


async def async_vlm_enrich(
    file_path: Path,
    image_pages: list[int],
    vector_store: Any,
    vlm_id: str,
    sparse_index: Any,
    document_registry: Any = None,
) -> dict[str, Any]:
    """
    Background task: run selected VLM on image-bearing pages.
    Prioritizes pages based on user attention (hits) if document_registry is provided.
    """
    if not image_pages:
        return {"status": "no_images", "chunks_added": 0, "vlm_id": vlm_id}

    # ── Priority Sorting ──────────────────────────────────────────
    if document_registry is not None:
        try:
            priority_pages = document_registry.get_page_priority(file_path.name)
            if priority_pages:
                # Move priority pages to the front, preserving relative order of the rest
                p_set = set(priority_pages)
                high_p = [p for p in image_pages if p in p_set]
                low_p = [p for p in image_pages if p not in p_set]
                # Further sort high_p by the actual hit count order from priority_pages
                high_p.sort(key=lambda x: priority_pages.index(x))
                image_pages = high_p + low_p
                logger.info(
                    "Priority Ingestion: moved %d attention-heavy pages to front for '%s'",
                    len(high_p),
                    file_path.name,
                )
        except Exception as e:
            logger.debug("Priority sorting failed (non-fatal): %s", e)

    from src.modules.ragforge_indexer import MEMORY_CEILING_PCT, MEMORY_CEILING_PCT_OLLAMA

    is_ollama_provider = "ollama" in vlm_id.lower()
    ceiling = MEMORY_CEILING_PCT_OLLAMA if is_ollama_provider else MEMORY_CEILING_PCT
    mem_pct = psutil.virtual_memory().percent

    if mem_pct >= ceiling:
        logger.warning(
            "VLM enrichment skipped for '%s' — memory at %.1f%% (ceiling: %.0f%%)",
            file_path.name,
            mem_pct,
            ceiling,
        )
        return {"status": "skipped_memory", "chunks_added": 0, "vlm_id": vlm_id}

    try:
        from src.modules.ragforge.vlm_provider import get_vlm_provider

        vlm = get_vlm_provider(vlm_id)
        if not vlm:
            return {"chunks_added": 0, "vlm_id": vlm_id, "last_error": "VLM provider unavailable."}

        import uuid as uuid_mod

        import fitz
        from langchain_core.documents import Document

        pdf_doc = fitz.open(str(file_path))
        extracted_chunks = []

        for page_num in image_pages:
            if page_num >= len(pdf_doc):
                continue

            if psutil.virtual_memory().percent >= ceiling:
                break

            page = pdf_doc[page_num]
            mat = fitz.Matrix(1.0, 1.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            try:
                prompt = (
                    "Analyze this page in detail. Extract data, labels, and logical relationships."
                )
                analysis = await vlm.analyze_image(img_bytes, prompt)
                from src.chat_contract import sanitize_output
                analysis = sanitize_output(analysis)

                if analysis and "NO_FIGURES" not in analysis:
                    extracted_chunks.append(
                        Document(
                            page_content=f"Visual Analysis of Page {page_num + 1}:\n{analysis}",
                            metadata={
                                "source": file_path.name,
                                "chunk_type": "vlm_analysis",
                                "page": page_num,
                                "chunk_id": str(uuid_mod.uuid4()),
                                "vlm_provider": vlm.id,
                            },
                        )
                    )
                elif "Ollama Error" in str(analysis):
                    logger.error("VLM provider is unreachable. Aborting VLM enrichment for remaining pages.")
                    break
            except Exception as e:
                logger.warning("VLM failed on page %d: %s", page_num + 1, e)
            finally:
                del pix, img_bytes

        pdf_doc.close()
        await vlm.unload_model()

        chunks_added = 0
        if extracted_chunks:
            source_name = file_path.name
            VLM_TIER_RANK = {"smolvlm-256m": 1, "florence-2": 2, "ollama-qwen3.5-9b": 3}
            current_rank = VLM_TIER_RANK.get(vlm.id, 1)

            # Logic for tier preservation
            if sparse_index is not None:
                try:
                    sparse_index.delete_vlm_chunks(source_name)
                    # Also clear from chroma
                    existing_chroma = vector_store.get(
                        where={"$and": [{"source": source_name}, {"chunk_type": "vlm_analysis"}]}
                    )
                    if existing_chroma and existing_chroma.get("ids"):
                        vector_store.delete(ids=existing_chroma["ids"])
                except Exception:
                    pass

            # Indexing new chunks
            await asyncio.to_thread(vector_store.add_documents, extracted_chunks)
            if sparse_index is not None:
                await asyncio.to_thread(sparse_index.add_documents, extracted_chunks)
            chunks_added = len(extracted_chunks)

        return {"chunks_added": chunks_added, "vlm_id": vlm.id, "last_error": None}

    except Exception as e:
        logger.error("VLM enrichment failed: %s", e)
        return {"chunks_added": 0, "vlm_id": vlm_id, "last_error": str(e)}
