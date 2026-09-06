import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import SemanticStructureChunker
from app.rag.embeddings import LocalEmbeddingEngine
from app.rag.vector_search import LocalVectorStore
from app.rag.bm25_search import LocalBM25Store
from app.rag.hybrid_search import HybridSearchEngine
from app.rag.reranker import LocalCrossEncoderReranker


def test_phase7():
    print(">>> 1/3 Initializing Hybrid Retriever...")
    embedder = LocalEmbeddingEngine()
    vstore = LocalVectorStore(storage_path="./data/qdrant_db", collection_name="hybrid_test_coll")
    bm25_store = LocalBM25Store(index_path="./data/hybrid_bm25.pkl")
    hybrid_engine = HybridSearchEngine(vstore, bm25_store, embedder)

    print("\n>>> 2/3 Retrieving top candidates via Hybrid RRF...")
    query = "What reranks candidates down to top context chunks?"
    initial_candidates = hybrid_engine.search(query=query, top_k=3)
    print(f"✓ Retrieved {len(initial_candidates)} candidates via Hybrid RRF.")

    print("\n>>> 3/3 Applying Cross-Encoder Neural Reranking...")
    reranker = LocalCrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    reranked_results = reranker.rerank(query=query, candidates=initial_candidates, top_n=2)

    assert len(reranked_results) > 0, "Reranking returned empty candidate list."
    
    print(f"\n✓ Top {len(reranked_results)} chunks after neural cross-attention reranking:")
    for rank, item in enumerate(reranked_results, 1):
        print(f"  [{rank}] Score: {item['rerank_score']:.4f} | ID: {item['metadata']['chunk_id']}")
        print(f"      Text: {item['text'][:85]}...")

    print("\n✓ Phase 7 Neural Cross-Encoder Reranker verified successfully.")

if __name__ == "__main__":
    test_phase7()