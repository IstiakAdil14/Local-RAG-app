import os 
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.rag.pipeline import AdvancedRAGPipeline

def test_full_pipeline():
    pipeline = AdvancedRAGPipeline(
        storage_path="./data/qdrant_db",
        collection_name="hybrid_test_coll",
        bm25_path="./data/hybrid_bm25.pkl"
    )
    query = "What technique is used to rerank top candidates down to 5 chunks?"
    print(f"\n--- Testing Query: '{query}' ---")
    response = pipeline.query(query, retrieval_candidates=3, top_n_rerank=2)

    print(f"\nAnswer:\n{response.answer}")
    print(f"\nCitations:\n{response.citations}")
    print(f"\nLatency Profiling:")
    print(f"  Retrieval (Hybrid RRF): {response.retrieval_latency_ms} ms")
    print(f"  Reranking (Cross-Encoder): {response.rerank_latency_ms} ms")
    print(f"  Generation (Qwen-0.5B): {response.generation_latency_ms} ms")
    print(f"  Total End-to-End: {response.total_latency_ms} ms")

    assert len(response.citations) > 0
    print("\n[OK] Full Advanced RAG Pipeline verified successfully.")
    
if __name__ == "__main__":
    test_full_pipeline()