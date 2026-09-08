import re
import time
from typing import Dict, Any, List, Optional

from app.rag.embeddings import LocalEmbeddingEngine
from app.rag.vector_search import LocalVectorStore
from app.rag.generator import LocalGenerator
from app.schemas.document import QueryResponse, Citation
from app.rag.bm25_search import LocalBM25Store
from app.rag.hybrid_search import HybridSearchEngine
from app.rag.reranker import LocalCrossEncoderReranker
from app.rag.query_classifier import QueryClassifier, QueryIntent
from app.rag.doc_intelligence import DocumentMetadataStore

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
            total_latency_ms=round(total_ms, 2),
            query_intent="fact",
            retrieval_strategy_used="baseline_vector"
        )

class AdvancedRAGPipeline:
    def __init__(
        self,
        storage_path: str = "./data/qdrant_db",
        collection_name: str = "rag_chunks",
        bm25_path: str = "./data/hybrid_bm25.pkl",
        metadata_store_path: str = "./data/doc_metadata.pkl",
        reranker_model: str = "BAAI/bge-reranker-base",
        generator_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        qdrant_url: str = None,
        qdrant_api_key: str = None
    ):
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.bm25_path = bm25_path
        self.metadata_store_path = metadata_store_path
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
        self._doc_metadata_store = None

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

    @property
    def doc_metadata_store(self):
        if self._doc_metadata_store is None:
            self._doc_metadata_store = DocumentMetadataStore(store_path=self.metadata_store_path)
        return self._doc_metadata_store

    def query(
        self,
        user_query: str,
        retrieval_candidates: int = 8,
        top_n_rerank: int = 4
    ) -> QueryResponse:
        total_start = time.time()

        # Step 1: Query Intent Detection
        intent = QueryClassifier.classify(user_query)
        strategy_used = intent.value

        latest_doc_meta = self.doc_metadata_store.get_latest_document()

        # --- ROUTE 1: METADATA QUERY ---
        if intent == QueryIntent.METADATA:
            t_ret_start = time.time()
            # Fetch page 1 chunk as context fallback if available
            p1_chunks = []
            if self.bm25_store.chunks:
                p1_chunks = [
                    {
                        "text": c.text,
                        "metadata": {
                            "document_id": c.metadata.document_id,
                            "document_name": c.metadata.document_name,
                            "page_number": c.metadata.page_number,
                            "section": c.metadata.section,
                            "chunk_id": c.metadata.chunk_id
                        }
                    }
                    for c in self.bm25_store.chunks if c.metadata.page_number == 1
                ]
            retrieval_ms = (time.time() - t_ret_start) * 1000.0
            rerank_ms = 0.0

            t_gen_start = time.time()
            answer = self.generator.generate_metadata_answer(
                query=user_query,
                doc_metadata=latest_doc_meta,
                fallback_chunks=p1_chunks
            )
            generation_ms = (time.time() - t_gen_start) * 1000.0
            total_ms = (time.time() - total_start) * 1000.0

            citations = []
            if latest_doc_meta:
                citations.append(Citation(
                    document_name=latest_doc_meta.document_name,
                    page_number=1,
                    section="Document Metadata / First Page",
                    chunk_id=f"{latest_doc_meta.document_id}_META"
                ))

            return QueryResponse(
                query=user_query,
                answer=answer,
                citations=citations,
                retrieval_latency_ms=round(retrieval_ms, 2),
                rerank_latency_ms=round(rerank_ms, 2),
                generation_latency_ms=round(generation_ms, 2),
                total_latency_ms=round(total_ms, 2),
                query_intent=intent.value,
                retrieval_strategy_used="metadata_layer"
            )

        # --- ROUTE 2: SUMMARY / LIST QUERY ---
        if intent in [QueryIntent.SUMMARY, QueryIntent.LIST]:
            t_ret_start = time.time()
            all_chunks = self.bm25_store.chunks
            if all_chunks:
                target_count = min(len(all_chunks), 15)
                if len(all_chunks) <= 15:
                    sampled = all_chunks
                else:
                    step = len(all_chunks) / float(target_count)
                    sampled = [all_chunks[int(i * step)] for i in range(target_count)]

                candidates = [
                    {
                        "text": c.text,
                        "metadata": {
                            "document_id": c.metadata.document_id,
                            "document_name": c.metadata.document_name,
                            "page_number": c.metadata.page_number,
                            "section": c.metadata.section,
                            "chunk_id": c.metadata.chunk_id
                        },
                        "score": 1.0,
                        "retrieval_method": "global_scan"
                    }
                    for c in sampled
                ]
            else:
                candidates = self.hybrid_engine.search(
                    query="document summary main topics overview",
                    top_k=max(12, retrieval_candidates),
                    candidate_pool=25
                )
            retrieval_ms = (time.time() - t_ret_start) * 1000.0

            t_rerank_start = time.time()
            reranked_chunks = sorted(
                candidates[:max(top_n_rerank, 10)],
                key=lambda c: (
                    c.get("metadata", {}).get("page_number", 0),
                    c.get("metadata", {}).get("chunk_id", "")
                )
            )
            rerank_ms = (time.time() - t_rerank_start) * 1000.0

            t_gen_start = time.time()
            if latest_doc_meta and latest_doc_meta.summary and len(latest_doc_meta.summary) > 40:
                answer = latest_doc_meta.summary
            else:
                answer = self.generator.summarize_document(reranked_chunks)
            generation_ms = (time.time() - t_gen_start) * 1000.0
            total_ms = (time.time() - total_start) * 1000.0

            citations = [
                Citation(
                    document_name=c.get("metadata", {}).get("document_name", "Unknown"),
                    page_number=c.get("metadata", {}).get("page_number", 1),
                    section=c.get("metadata", {}).get("section", "General"),
                    chunk_id=c.get("metadata", {}).get("chunk_id", "N/A")
                )
                for c in reranked_chunks
            ]

            return QueryResponse(
                query=user_query,
                answer=answer,
                citations=citations,
                retrieval_latency_ms=round(retrieval_ms, 2),
                rerank_latency_ms=round(rerank_ms, 2),
                generation_latency_ms=round(generation_ms, 2),
                total_latency_ms=round(total_ms, 2),
                query_intent=intent.value,
                retrieval_strategy_used="document_summary_scan"
            )

        # --- ROUTE 3: FACT / COMPARISON (Hybrid RAG + 5-Tier Fallback Cascade) ---
        t_ret_start = time.time()
        effective_candidates = max(retrieval_candidates, 8)
        effective_top_n = max(top_n_rerank, 4)

        # Tier 1: Hybrid RAG Search
        candidates = self.hybrid_engine.search(
            query=user_query,
            top_k=effective_candidates,
            candidate_pool=max(20, effective_candidates * 2)
        )
        retrieval_ms = (time.time() - t_ret_start) * 1000.0

        t_rerank_start = time.time()
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
            max_tokens=250
        )
        strategy_used = "hybrid_rerank"

        # Tier 2 Fallback: First Page / Document Title check if answer is not specified
        if "does not specify" in answer.lower() or "not specify" in answer.lower():
            if any(k in user_query.lower() for k in ["title", "called", "subject", "course", "header", "author"]):
                p1_chunks = [c for c in (self.bm25_store.chunks or []) if c.metadata.page_number == 1]
                if p1_chunks or latest_doc_meta:
                    answer = self.generator.generate_metadata_answer(user_query, latest_doc_meta, [
                        {"text": c.text, "metadata": {"document_name": c.metadata.document_name, "page_number": c.metadata.page_number, "section": c.metadata.section, "chunk_id": c.metadata.chunk_id}}
                        for c in p1_chunks
                    ])
                    if "does not specify" not in answer.lower():
                        strategy_used = "fallback_tier2_metadata"

        # Tier 3 Fallback: Keyword search across BM25 if still not specified
        if "does not specify" in answer.lower():
            bm25_results = self.bm25_store.search(user_query, top_k=6)
            if bm25_results:
                kw_answer = self.generator.generate_grounded_answer(user_query, bm25_results, max_tokens=250)
                if "does not specify" not in kw_answer.lower():
                    answer = kw_answer
                    reranked_chunks = bm25_results
                    strategy_used = "fallback_tier3_bm25_keyword"

        # Tier 4 Fallback: Document Overview Context if semi-broad query still unspecified
        if "does not specify" in answer.lower() and latest_doc_meta and latest_doc_meta.summary:
            if any(k in user_query.lower() for k in ["topic", "content", "field", "area", "dataset", "purpose"]):
                answer = f"Based on the document overview: {latest_doc_meta.summary}"
                strategy_used = "fallback_tier4_doc_summary"

        # Answer Verification Layer
        from app.rag.answer_verifier import AnswerVerifier
        answer = AnswerVerifier.verify(
            query=user_query,
            raw_answer=answer,
            context_chunks=reranked_chunks,
            doc_metadata=latest_doc_meta
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
            total_latency_ms=round(total_ms, 2),
            query_intent=intent.value,
            retrieval_strategy_used=strategy_used
        )


        
