import asyncio
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def main():
    loader = PyPDFLoader("data/LiveFolder/HA - 13 LOADING AND STABILITY INFORMATION BOOKLET.pdf")
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} pages.")

    found = False
    for i, doc in enumerate(raw_docs):
        if "7.47" in doc.page_content:
            found = True
            print(f"\n--- MATCH ON PAGE {doc.metadata.get('page', 'Unknown')} ---")
            idx = doc.page_content.find("7.47")
            start = max(0, idx - 500)
            end = min(len(doc.page_content), idx + 500)
            print("Content snippet (context around 7.47):")
            print(doc.page_content[start:end])
    
    if not found:
        print("Could not find '7.47' in any page. The text extractor may have failed, or it's an image/scan.")

if __name__ == "__main__":
    main()
