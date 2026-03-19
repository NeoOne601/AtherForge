import asyncio
from pathlib import Path
from src.modules.ragforge_indexer import load_document
import logging

logging.basicConfig(level=logging.INFO)

def main():
    path = Path("data/LiveFolder/HA - 13 LOADING AND STABILITY INFORMATION BOOKLET.pdf")
    docs, image_pages = load_document(path)
    print(f"Extracted {len(docs)} text chunks. {len(image_pages)} pages queued for VLM.")

    found = False
    for i, doc in enumerate(docs):
        if "7.47" in doc.page_content:
            found = True
            print(f"\n--- MATCH IN CHUNK {i} ---")
            print(f"Metadata: {doc.metadata}")
            idx = doc.page_content.find("7.47")
            start = max(0, idx - 500)
            end = min(len(doc.page_content), idx + 1000)
            print("Content snippet (context around 7.47):")
            print(doc.page_content[start:end])
    
    if not found:
        print("Could not find '7.47' in any chunk. The text extractor may have failed, or it's an image/scan.")

if __name__ == "__main__":
    main()
