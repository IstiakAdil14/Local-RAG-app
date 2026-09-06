import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import SemanticStructureChunker
from app.rag.bm25_search import LocalBM25Store

def test_phase5():
    print(">>> 1/3 Parsing sample text for BM25 indexing...")
    sample_file = "data/raw/sample_test.txt"
    pages = DocumentParser.parse(sample_file)
    chunker = SemanticStructureChunker(max_chunk_size=20)
    chunks = chunker.chunk(pages, doc_id="DOC_BM25", doc_name="sample_test.txt")
    print(f"✓ Parsed and created {len(chunks)} chunks.")

    print("\n>>> 2/3 Building and persisting BM25 index...")
    bm25_store = LocalBM25Store(index_path="./data/test_bm25.pkl")
    bm25_store.index_chunks(chunks)
    print("✓ BM25 index built and saved.")

    print("\n>>> 3/3 Testing keyword search query...")
    query = "cross-encoder"
    results = bm25_store.search(query, top_k=2)

    assert len(results) > 0, "No results returned from BM25 search."
    print(f"✓ Returned {len(results)} hit(s) for query '{query}':")
    print(f"  Score: {results[0]['score']:.4f}")
    print(f"  Snippet: {results[0]['text'][:80]}...")
    print(f"  Metadata: {results[0]['metadata']}")

    print("\n✓ Phase 5 BM25 search engine verified successfully.")

if __name__ == "__main__":
    test_phase5()