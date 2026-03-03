import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the exact same vector store
chroma_path = Path("/Users/macuser/AtherForge/data/chroma")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=str(chroma_path), 
    embedding_function=embeddings
)

docs = vector_store.get()
print(f"Total documents in Chroma DB: {len(docs['ids'])}")

for i in range(len(docs['ids'])):
    print(f"\n--- Doc {i} ---")
    print(f"ID: {docs['ids'][i]}")
    print(f"Metadata: {docs['metadatas'][i]}")
    print(f"Content preview: {docs['documents'][i][:100]}...")

# Test similarity search
print("\n--- Testing similarity_search ---")
query = "summarize the document"
results = vector_store.similarity_search(query, k=5)
print(f"Results without filter: {len(results)}")

filter_dict = {"source": "Industry Claims & SLA Benchmarks.pdf"}
results_filtered = vector_store.similarity_search(query, k=5, filter=filter_dict)
print(f"Results WITH filter: {len(results_filtered)}")
for res in results_filtered:
    print(res.metadata)

