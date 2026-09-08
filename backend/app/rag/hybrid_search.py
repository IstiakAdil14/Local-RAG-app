from typing import List, Dict, Any
from app.rag.embeddings import LocalEmbeddingEngine
from app.rag.vector_search import LocalVectorStore
from app.rag.bm25_search import LocalBM25Store

class HybridSearchEngine:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        bm25_store: LocalBM25Store,
        embedder: LocalEmbeddingEngine,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder
        self.rrf_k = rrf_k
    
    def search(self, query: str, top_k: int = 5, candidate_pool: int = 15) -> List[Dict[str, Any]]:
        pool_size = max(top_k, candidate_pool)
        q_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(q_vec, top_k=pool_size)
        sparse_results = self.bm25_store.search(query, top_k=pool_size)

        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        
        for rank, item in enumerate(dense_results, start=1):
            chunk_id = item["metadata"]["chunk_id"]   
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = item

        for rank, item in enumerate(sparse_results, start=1):
            chunk_id = item["metadata"]["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = item

        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_fused_results = []
        seen_text_signatures = set()

        for cid in sorted_chunk_ids:
            entry = chunk_lookup[cid].copy()
            text_sig = entry.get("text", "").strip().lower()
            
            # Filter out exact duplicate chunk contents
            if text_sig and text_sig in seen_text_signatures:
                continue
            seen_text_signatures.add(text_sig)

            entry["score"] = round(rrf_scores[cid], 6)
            entry["retrieval_method"] = "hybrid_rrf"
            final_fused_results.append(entry)

            if len(final_fused_results) >= top_k:
                break
        
        return final_fused_results