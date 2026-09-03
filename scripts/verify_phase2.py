import os
import sys

# Add backend directory to Python module path
sys.path.insert(0, os.path.abspath("backend"))

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import FixedSizeChunker, SemanticStructureChunker

def test_phase2():
    os.makedirs("data/raw", exist_ok=True)
    sample_file = "data/raw/sample_test.txt"
    
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(
            "1. EXECUTIVE SUMMARY\n"
            "This project evaluates fully local Retrieval-Augmented Generation.\n\n"
            "2. RETRIEVAL DESIGN\n"
            "We integrate dense vector representations using BGE-M3 and sparse lexical frequencies using BM25.\n"
            "A neural cross-encoder then reranks top candidates down to the top 5 context chunks."
        )
    
    print(">>> 1/3 Parsing document...")
    pages = DocumentParser.parse(sample_file)
    print(f"✓ Parsed {len(pages)} page(s). Section: '{pages[0]['section']}'")

    print(">>> 2/3 Testing Fixed-Size Chunker...")
    fixed_chunker = FixedSizeChunker(chunk_size=12, chunk_overlap=3)
    fixed_chunks = fixed_chunker.chunk(pages, doc_id="DOC_001", doc_name="sample_test.txt")
    print(f"✓ Created {len(fixed_chunks)} fixed chunks.")

    print(">>> 3/3 Testing Semantic Chunker...")
    semantic_chunker = SemanticStructureChunker(max_chunk_size=18)
    semantic_chunks = semantic_chunker.chunk(pages, doc_id="DOC_001", doc_name="sample_test.txt")
    print(f"✓ Created {len(semantic_chunks)} semantic chunks.")
    print(f"  Sample Chunk Metadata: {semantic_chunks[0].metadata.model_dump()}")
    
    print("\n✓ Phase 2 ingestion and chunking verified successfully.")

if __name__ == "__main__":
    test_phase2()