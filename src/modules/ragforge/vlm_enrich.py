from __future__ import annotations
import structlog
import psutil
from typing import Any
from pathlib import Path
from src.utils import safe_create_task

logger = structlog.get_logger("aetherforge.vlm_enrich")

async def async_vlm_enrich(
    file_path: Path,
    image_pages: list[int],
    vector_store: Any,
    vlm_id: str,
    sparse_index: Any,
) -> None:
    """
    Background task: run selected VLM on image-bearing pages.
    """
    if not image_pages:
        return

    from src.modules.ragforge_indexer import MEMORY_CEILING_PCT, MEMORY_CEILING_PCT_OLLAMA
    is_ollama_provider = "ollama" in vlm_id.lower()
    ceiling = MEMORY_CEILING_PCT_OLLAMA if is_ollama_provider else MEMORY_CEILING_PCT
    mem_pct = psutil.virtual_memory().percent
    
    if mem_pct >= ceiling:
        logger.warning(
            "VLM enrichment skipped for '%s' — memory at %.1f%% (ceiling: %.0f%%)",
            file_path.name, mem_pct, ceiling
        )
        return
        
    try:
        from src.modules.ragforge.vlm_provider import get_vlm_provider
        vlm = get_vlm_provider(vlm_id)
        if not vlm:
            return
            
        import fitz
        from langchain_core.documents import Document
        import uuid as uuid_mod
        
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
                prompt = "Analyze this page in detail. Extract data, labels, and logical relationships."
                analysis = await vlm.analyze_image(img_bytes, prompt)
                
                if analysis and "NO_FIGURES" not in analysis:
                    extracted_chunks.append(Document(
                        page_content=f"Visual Analysis of Page {page_num + 1}:\n{analysis}",
                        metadata={
                            "source": file_path.name,
                            "chunk_type": "vlm_analysis",
                            "page": page_num,
                            "chunk_id": str(uuid_mod.uuid4()),
                            "vlm_provider": vlm.id
                        }
                    ))
            except Exception as e:
                logger.warning("VLM failed on page %d: %s", page_num + 1, e)
            finally:
                del pix, img_bytes
            
        pdf_doc.close()
        await vlm.unload_model()
        
        if extracted_chunks:
            source_name = file_path.name
            VLM_TIER_RANK = {"smolvlm-256m": 1, "florence-2": 2, "ollama-qwen3.5-9b": 3}
            current_rank = VLM_TIER_RANK.get(vlm.id, 1)
            
            # Logic for tier preservation
            if sparse_index is not None:
                try:
                    sparse_index.delete_vlm_chunks(source_name)
                    # Also clear from chroma
                    existing_chroma = vector_store.get(where={
                        "$and": [{"source": source_name}, {"chunk_type": "vlm_analysis"}]
                    })
                    if existing_chroma and existing_chroma.get("ids"):
                        vector_store.delete(ids=existing_chroma["ids"])
                except Exception:
                    pass
            
            # Indexing new chunks
            await asyncio.to_thread(vector_store.add_documents, extracted_chunks)
            if sparse_index is not None:
                for doc in extracted_chunks:
                    sparse_index.insert_vlm_chunk(doc)
                    
    except Exception as e:
        logger.error("VLM enrichment failed: %s", e)
