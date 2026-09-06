import time
from typing import Dict, Any, List
from app.rag.embeddings import LocalEmbeddingEngine
from app.rag.vector_search import LocalVectorStore
from app.rag.generator import LocalGenerator
from app.schemas.document import QueryResponse, Citation
from app.rag.bm25_search import LocalBM25Store
from app.rag.hybrid_search import HybridSearchEngine
from app.rag.reranker import LocalCrossEncoderReranker
class BaseLineRAGPipeline:
    def __init__(self, storage_path: str = "./data/qdrant_db", collection_name: str = "rag_chunks"):
        self.embedder = LocalEmbeddingEngine()
        self.vector_store = LocalVectorStore(storage_path=storage_path, collection_name=collection_name)
        self.generator = LocalGenerator()

    def query(self, user_query: str, top_k: int = 3) -> QueryResponse:
        total_start = time.time()

        # Step 1: Retrieval
        t_ret_start = time.time()
        query_vector = self.embedder.embed_query(user_query)
        retrieved_chunks = self.vector_store.search(query_vector, top_k=top_k)
        retrieval_ms = (time.time() - t_ret_start) * 1000.0

        # Step 2: Rerank (Phase 4 baseline has no reranker yet)
        rerank_ms = 0.0

        # Step 3: Generation
        t_gen_start = time.time()
        answer = self.generator.generate_grounded_answer(user_query, retrieved_chunks)
        generation_ms = (time.time() - t_gen_start) * 1000.0

        total_ms = (time.time() - total_start) * 1000.0

        # Step 4: Citations Extraction
        citations: List[Citation] = []
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            citations.append(
                Citation(
                    document_name=meta.get("document_name", "Unknown"),
                    page_number=meta.get("page_number", 1),
                    section=meta.get("section", "General"),
                    chunk_id=meta.get("chunk_id", "N/A")
                )
            )

        return QueryResponse(
            query=user_query,
            answer=answer,
            citations=citations,
            retrieval_latency_ms=round(retrieval_ms, 2),
            rerank_latency_ms=round(rerank_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
            total_latency_ms=round(total_ms, 2)
        )

class AdvancedRAGPipeline:
    def __init__(
        self,
        storage_path: str = "./data/qdrant_db",
        collection_name: str = "rag_chunks",
        bm25_path: str = "./data/hybrid_bm25.pkl",
        reranker_model: str = "BAAI/bge-reranker-base",
        generator_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        print(">>> Initializing Advanced RAG Components...")
        self.embedder = LocalEmbeddingEngine()
        self.vector_store = LocalVectorStore(storage_path=storage_path, collection_name=collection_name)
        self.bm25_store = LocalBM25Store(index_path=bm25_path)

        self.hybrid_engine = HybridSearchEngine(
            vector_store=self.vector_store, 
            bm25_store=self.bm25_store, 
            embedder=self.embedder,
            rrf_k=60
            )
        self.reranker = LocalCrossEncoderReranker(model_name=reranker_model)
        self.generator = LocalGenerator(model_id=generator_model)
    
    def query(
        self,
        user_query:str,
        retrieval_candidates:int=3,
        top_n_rerank: int=3
    ) -> QueryResponse:
        total_start = time.time()
        candidates = self.hybrid_engine.search(
            query=user_query,
            top_k=retrieval_candidates,
        )
        retrieval_ms = (time.time() - total_start) * 1000.0
        
        t_rerank_start = time.time()
        reranked_chunks = self.reranker.rerank(
            query=user_query,
            candidates=candidates,
            top_n=top_n_rerank
        )
        rerank_ms = (time.time() - t_rerank_start) * 1000.0

        t_gen_start = time.time()
        answer = self.generator.generate_grounded_answer(user_query, reranked_chunks)
        generation_ms = (time.time() - t_gen_start) * 1000.0

        total_ms = (time.time() - total_start) * 1000.0

        citations: List[Citation] = []
        for c in reranked_chunks:
            meta = c.get("metadata", {})
            citations.append(
                Citation(
                    document_name=meta.get("document_name", "Unknown"),
                    page_number=meta.get("page_number", 1),
                    section=meta.get("section", "General"),
                    chunk_id=meta.get("chunk_id", "N/A")
                )
            )

        return QueryResponse(
            query=user_query,
            answer=answer,
            citations=citations,
            retrieval_latency_ms=round(retrieval_ms, 2),
            rerank_latency_ms=round(rerank_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
            total_latency_ms=round(total_ms, 2)
        )

        
