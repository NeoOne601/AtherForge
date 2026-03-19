import asyncio
from pathlib import Path
from src.modules.ragforge.parsers import PdfPlumberDocumentParser

async def main():
    parser = PdfPlumberDocumentParser()
    path = Path("data/LiveFolder/HA - 13 LOADING AND STABILITY INFORMATION BOOKLET.pdf")
    docs = await parser.parse(path)
    print(f"Parsed {len(docs)} documents.")

    found = False
    for i, doc in enumerate(docs):
        if "7.47" in doc.page_content:
            found = True
            print(f"\n--- MATCH IN CHUNK {i} ---")
            print(f"Metadata: {doc.metadata}")
            idx = doc.page_content.find("7.47")
            start = max(0, idx - 200)
            end = min(len(doc.page_content), idx + 1000)
            print("Content snippet (context around 7.47):")
            print(doc.page_content[start:end])
    
    if not found:
        print("Could not find '7.47' in any chunk. The text extractor may have failed, or it's an image/scan.")

if __name__ == "__main__":
    asyncio.run(main())
