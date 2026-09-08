import re
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
    def __init__(
        self,
        storage_path: str = "./data/qdrant_db",
        collection_name: str = "rag_chunks",
        qdrant_url: str = None,
        qdrant_api_key: str = None
    ):
        self.embedder = LocalEmbeddingEngine()
        self.vector_store = LocalVectorStore(
            storage_path=storage_path,
            collection_name=collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key
        )
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
        generator_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        qdrant_url: str = None,
        qdrant_api_key: str = None
    ):
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.bm25_path = bm25_path
        self.reranker_model = reranker_model
        self.generator_model = generator_model
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

        self._embedder = None
        self._vector_store = None
        self._bm25_store = None
        self._hybrid_engine = None
        self._reranker = None
        self._generator = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = LocalEmbeddingEngine()
        return self._embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = LocalVectorStore(
                storage_path=self.storage_path,
                collection_name=self.collection_name,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
        return self._vector_store

    @property
    def bm25_store(self):
        if self._bm25_store is None:
            self._bm25_store = LocalBM25Store(index_path=self.bm25_path)
        return self._bm25_store

    @property
    def hybrid_engine(self):
        if self._hybrid_engine is None:
            self._hybrid_engine = HybridSearchEngine(
                vector_store=self.vector_store, 
                bm25_store=self.bm25_store, 
                embedder=self.embedder,
                rrf_k=60
            )
        return self._hybrid_engine

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = LocalCrossEncoderReranker(model_name=self.reranker_model)
        return self._reranker

    @property
    def generator(self):
        if self._generator is None:
            self._generator = LocalGenerator(model_id=self.generator_model)
        return self._generator

    def _is_global_query(self, query: str) -> bool:
        q_lower = query.lower()
        patterns = [
            r"\b(list|show|explain|give)\b.*\b(every|all)\b",
            r"\b(every|all)\b.*\b(cell|cells|section|sections|page|pages|doc|document)\b",
            r"\bsummarize\b",
            r"\bsummary\b",
            r"\blist all\b",
            r"\blist every\b",
            r"\ball cells\b",
            r"\bevery cell\b",
            r"\boverview\b"
        ]
        return any(re.search(p, q_lower) for p in patterns)
    
    def query(
        self,
        user_query: str,
        retrieval_candidates: int = 8,
        top_n_rerank: int = 4
    ) -> QueryResponse:
        total_start = time.time()

        is_global = self._is_global_query(user_query)
        effective_candidates = max(retrieval_candidates, 12) if is_global else max(retrieval_candidates, 8)
        effective_top_n = max(top_n_rerank, 5) if is_global else max(top_n_rerank, 4)

        candidates = self.hybrid_engine.search(
            query=user_query,
            top_k=effective_candidates,
            candidate_pool=max(15, effective_candidates * 2)
        )
        retrieval_ms = (time.time() - total_start) * 1000.0
        
        t_rerank_start = time.time()
        if is_global:
            # For global aggregation queries, order candidate chunks sequentially by page and chunk_id
            reranked_chunks = sorted(
                candidates[:effective_top_n],
                key=lambda c: (
                    c.get("metadata", {}).get("page_number", 0),
                    c.get("metadata", {}).get("chunk_id", "")
                )
            )
        else:
            reranked_chunks = self.reranker.rerank(
                query=user_query,
                candidates=candidates,
                top_n=effective_top_n
            )
        rerank_ms = (time.time() - t_rerank_start) * 1000.0

        t_gen_start = time.time()
        answer = self.generator.generate_grounded_answer(
            user_query, 
            reranked_chunks,
            max_tokens=400 if is_global else 250
        )
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

        
