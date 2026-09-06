import os 
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import SemanticStructureChunker
from app.rag.embeddings import LocalEmbeddingEngine
from app.rag.vector_search import LocalVectorStore
from app.rag.bm25_search import LocalBM25Store
from app.rag.hybrid_search import HybridSearchEngine

def test_phase6():
    print(">>> 1/3 Preparing shared chunks for Dense and BM25 stores...")
    sample_file = "data/raw/sample_test.txt"
    pages = DocumentParser.parse(sample_file)
    chunker = SemanticStructureChunker(max_chunk_size=20)
    chunks = chunker.chunk(pages, doc_id="DOC_HYBRID", doc_name="sample_test.txt")

    embedder = LocalEmbeddingEngine()
    embeddings = embedder.embed_texts([c.text for c in chunks])

    # Index into vector store
    vstore = LocalVectorStore(storage_path="./data/qdrant_db", collection_name="hybrid_test_coll")
    vstore.index_chunks(chunks, embeddings)

    # Index into BM25 store
    bm25_store = LocalBM25Store(index_path="./data/hybrid_bm25.pkl")
    bm25_store.index_chunks(chunks)
    print(f"✓ Synchronized {len(chunks)} chunks across Vector and BM25 stores.")

    print("\n>>> 2/3 Executing Hybrid Search with RRF...")
    hybrid_engine = HybridSearchEngine(
        vector_store=vstore,
        bm25_store=bm25_store,
        embedder=embedder,
        rrf_k=60
    )

    query = "neural cross-encoder reranking"
    results = hybrid_engine.search(query=query, top_k=3)

    assert len(results) > 0, "No results returned from Hybrid search."

    print(f"✓ Hybrid search returned {len(results)} hit(s) for query: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] RRF Score: {r['score']} | Chunk: {r['metadata']['chunk_id']}")
        print(f"      Text: {r['text'][:75]}...")

    print("\n✓ Phase 6 Hybrid Search (RRF) verified successfully.")

if __name__ == "__main__":
    test_phase6()