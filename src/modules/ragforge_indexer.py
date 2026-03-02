# AetherForge v1.0 — src/modules/ragforge_indexer.py
# ─────────────────────────────────────────────────────────────────
# Document ingestion and vectorization for RAGForge.
# Supports PDF, TXT, MD, CSV. Uses entirely local embeddings.
# ─────────────────────────────────────────────────────────────────
import logging
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("aetherforge.ragforge_indexer")

def load_document(filepath: Path) -> list[Document]:
    """Load a document into LangChain Document objects based on file extension."""
    ext = filepath.suffix.lower()
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(filepath))
            return loader.load()
        elif ext == ".csv":
            loader = CSVLoader(str(filepath))
            return loader.load()
        elif ext in [".txt", ".md"]:
            loader = TextLoader(str(filepath), encoding="utf-8")
            return loader.load()
        else:
            logger.warning(f"Unsupported file type '{ext}' for file {filepath.name}. Falling back to TextLoader.")
            loader = TextLoader(str(filepath), encoding="utf-8")
            return loader.load()
    except Exception as e:
        logger.error(f"Error loading document {filepath}: {e}")
        return []

def index_document(filepath: Path, vector_store) -> int:
    """
    Parse the document, split into chunks, and add to the provided local Chroma vector store.
    Returns the number of chunks added.
    """
    logger.info(f"RAGForge indexing document: {filepath.name}")
    
    # 1. Load document
    docs = load_document(filepath)
    if not docs:
        logger.error(f"Failed to extract any text from {filepath.name}")
        return 0
        
    # Add metadata to identify source in retrieval
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = filepath.name
        else:
            # Overwrite absolute path with just the filename for cleaner citations
            doc.metadata["source"] = Path(doc.metadata["source"]).name

    # 2. Split into chunks
    # 1000 characters with 200 overlap is standard for LLM RAG pipelines
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    
    if not chunks:
        logger.warning(f"Document {filepath.name} yielded 0 chunks.")
        return 0
        
    # 3. Insert into Vector Database
    logger.info(f"Adding {len(chunks)} chunks to ChromaDB...")
    vector_store.add_documents(chunks)
    logger.info(f"Successfully indexed '{filepath.name}' ({len(chunks)} chunks).")
    
    return len(chunks)
