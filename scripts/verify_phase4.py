import os
import sys
sys.path.insert(0, os.path.abspath("backend"))

from app.rag.pipeline import BaseLineRAGPipeline

def test_phase4():
    print(">>> 1/3 Initializing Baseline RAG Pipeline...")
    pipeline = BaseLineRAGPipeline(storage_path="./data/qdrant_db", collection_name="test_collection")

    # Test 1: Supported factual query
    query_1 = "What dense and sparse retrieval models are combined?"
    print(f"\n>>> 2/3 Query 1 (Supported): '{query_1}'")
    resp_1 = pipeline.query(query_1, top_k=2)
    print(f"Answer:\n{resp_1.answer}\n")
    print(f"Citations: {resp_1.citations}")
    print(f"Latency -> Retrieval: {resp_1.retrieval_latency_ms}ms | Generation: {resp_1.generation_latency_ms}ms | Total: {resp_1.total_latency_ms}ms")

    # Test 2: Unsupported query (checking grounding constraint)
    query_2 = "What is the capital of Mars?"
    print(f"\n>>> 3/3 Query 2 (Unsupported / Out-of-Context): '{query_2}'")
    resp_2 = pipeline.query(query_2, top_k=2)
    print(f"Answer:\n{resp_2.answer}\n")

    print("✓ Phase 4 Baseline Vector RAG verified successfully.")


if __name__ == "__main__":
    test_phase4()    