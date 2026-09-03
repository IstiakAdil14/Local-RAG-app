import os
import json
import time
import sys
from datetime import datetime

# Reconfigure stdout to support UTF-8 (e.g. Bangla characters) on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.retrieval.service import RetrievalService
from app.services.vector_store.service import QdrantService
from app.services.embeddings.service import EmbeddingService

def run_experiments():
    print("==================================================")
    print("RUNNING RETRIEVAL EXPERIMENTS")
    print("==================================================")
    
    # Initialize services
    qdrant_svc = QdrantService()
    embedding_svc = EmbeddingService()
    retrieval_svc = RetrievalService(qdrant_service=qdrant_svc, embedding_service=embedding_svc)
    
    # Ensure collection exists and there is at least one document indexed
    base_dir = qdrant_svc.base_dir
    chunks_dir = os.path.join(qdrant_svc.processed_dir, "chunks")
    
    if not os.path.exists(chunks_dir):
        print(f"Chunks directory does not exist: {chunks_dir}")
        print("Please upload and chunk a document first using the frontend or script.")
        return
        
    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith(".json")]
    if not chunk_files:
        print("No processed chunk files found on disk. Please process a document first.")
        return
        
    print(f"Found {len(chunk_files)} chunk files on disk.")
    
    # Find a document to use
    sample_file = chunk_files[0]
    # Format of chunk filename: {document_id}_chunks_{strategy}.json
    parts = sample_file.split("_chunks_")
    if len(parts) != 2:
        print(f"Skipping incorrectly formatted file: {sample_file}")
        return
    document_id = parts[0]
    strategy = parts[1].replace(".json", "")
    
    print(f"Using document_id: {document_id} with strategy: {strategy}")
    
    # Verify index status in Qdrant
    status = qdrant_svc.get_document_status(document_id)
    if not status.indexed:
        print(f"Document {document_id} not indexed in Qdrant. Indexing it now...")
        # Check if embeddings file exists first
        emb_file = os.path.join(qdrant_svc.embeddings_dir, f"{document_id}_embeddings_{strategy}.json")
        if not os.path.exists(emb_file):
            print(f"Embeddings file not found: {emb_file}. Generating embeddings first...")
            embedding_svc.generate_embeddings(document_id, strategy)
        qdrant_svc.index_document(document_id, strategy)
        print("Indexing completed successfully!")
    else:
        print(f"Document {document_id} is already indexed ({status.vector_count} vectors).")
        
    # Sample queries (English, Bangla, Multilingual, and non-matching)
    sample_queries = [
        "test document in English",
        "Bangla textএটি একটি পরীক্ষা",
        "এটি একটি পরীক্ষা",
        "learning python and javascript",
        "machine learning algorithms"
    ]
    
    results = []
    
    print("\nExecuting queries against indexed collection...")
    for i, query in enumerate(sample_queries):
        print(f"\nQuery {i+1}: '{query}'")
        try:
            start_t = time.time()
            # Perform search (Top-K = 5, no doc filter to test global search)
            search_res = retrieval_svc.retrieve_chunks(query, top_k=5)
            elapsed = time.time() - start_t
            
            # Compute evaluation metrics
            scores = [item.score for item in search_res.results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            
            retrieved_docs_meta = [
                {
                    "document_id": item.document_id,
                    "chunk_index": item.chunk_index,
                    "score": item.score,
                    "snippet": item.text[:100] + "..." if len(item.text) > 100 else item.text
                }
                for item in search_res.results
            ]
            
            query_record = {
                "query": query,
                "top_k": 5,
                "search_time_seconds": search_res.search_time_seconds,
                "retrieved_count": search_res.retrieved_count,
                "retrieved_documents": retrieved_docs_meta,
                "metrics": {
                    "average_score": avg_score,
                    "maximum_score": max_score,
                    "total_retrieval_time_seconds": elapsed
                }
            }
            results.append(query_record)
            
            print(f"  -> Retrieved {search_res.retrieved_count} chunks in {search_res.search_time_seconds:.4f}s")
            print(f"  -> Max Score: {max_score:.4f} | Avg Score: {avg_score:.4f}")
            if retrieved_docs_meta:
                print(f"  -> Top Match: '{retrieved_docs_meta[0]['snippet']}'")
        except Exception as e:
            print(f"  -> Query failed: {str(e)}")
            
    # Save experiment report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, "experiments", "retrieval")
    os.makedirs(output_dir, exist_ok=True)
    
    report_filename = f"retrieval_exp_{timestamp}.json"
    report_path = os.path.join(output_dir, report_filename)
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "document_id": document_id,
        "strategy": strategy,
        "queries": results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n==================================================")
    print(f"Experiment results saved to:")
    print(f"{report_path}")
    print(f"==================================================")

if __name__ == "__main__":
    run_experiments()
