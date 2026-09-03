import os
import json
import time
from typing import List, Dict, Any, Optional

from qdrant_client.http import models

from app.schemas.retrieval import RetrievedChunk, SearchResponse
from app.services.embeddings.service import EmbeddingService
from app.services.vector_store.service import QdrantService

class RetrievalService:
    def __init__(self, qdrant_service: Optional[QdrantService] = None, embedding_service: Optional[EmbeddingService] = None):
        # Base directory resolution
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        self.processed_dir = os.path.join(self.base_dir, "documents", "processed")
        self.chunks_dir = os.path.join(self.processed_dir, "chunks")
        
        self.qdrant_service = qdrant_service or QdrantService()
        self.embedding_service = embedding_service or EmbeddingService()

    def retrieve_chunks(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> SearchResponse:
        """
        Embeds the query, queries Qdrant database, and retrieves matching chunks.
        """
        # 1. Input Validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if top_k > 20:
            raise ValueError("top_k cannot exceed 20.")

        start_time = time.time()

        # 2. Generate Embedding
        embed_start = time.time()
        try:
            query_vector = self.embedding_service.embed_query(query)
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
        embed_time = time.time() - embed_start

        # 3. Query Qdrant
        search_start = time.time()
        try:
            client = self.qdrant_service._get_client()
        except Exception as e:
            raise RuntimeError(f"Could not connect to Qdrant: {str(e)}")

        collection_name = self.qdrant_service.collection_name

        # Ensure collection exists and database is accessible
        try:
            collections = client.get_collections().collections
            col_names = [c.name for c in collections]
            if collection_name not in col_names:
                raise ValueError(f"Collection '{collection_name}' does not exist. Please index a document first.")
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise RuntimeError(f"Qdrant database connection error: {str(e)}")

        # Build filter if document_id is provided
        query_filter = None
        if document_id:
            # Basic validation of document_id format (no directory traversals)
            if not document_id.replace("_", "").replace("-", "").replace(".", "").isalnum():
                raise ValueError("Invalid document ID format.")
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )

        try:
            search_response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            search_results = search_response.points
        except Exception as e:
            raise RuntimeError(f"Qdrant search failed: {str(e)}")

        search_time = time.time() - search_start

        # 4. Resolve Chunk Texts from Disk
        retrieved_chunks: List[RetrievedChunk] = []
        chunks_cache: Dict[str, Dict[str, Dict[str, Any]]] = {} # dict key strategy_doc_id -> {chunk_id: chunk_data}

        for point in search_results:
            payload = point.payload or {}
            doc_id = payload.get("document_id")
            strategy = payload.get("strategy", "fixed")
            chunk_id = payload.get("chunk_id")
            chunk_index = payload.get("chunk_index")

            chunk_text = ""
            if doc_id and strategy:
                cache_key = f"{doc_id}_{strategy}"
                if cache_key not in chunks_cache:
                    chunks_filepath = os.path.join(self.chunks_dir, f"{doc_id}_chunks_{strategy}.json")
                    if os.path.exists(chunks_filepath):
                        try:
                            with open(chunks_filepath, "r", encoding="utf-8") as f:
                                chunks_data = json.load(f)
                                # Build dictionary keyed by chunk_id for faster lookup
                                chunks_cache[cache_key] = {c.get("chunk_id"): c for c in chunks_data}
                        except Exception:
                            chunks_cache[cache_key] = {}
                    else:
                        chunks_cache[cache_key] = {}

                # Look up chunk text
                chunk_record = chunks_cache[cache_key].get(chunk_id)
                if chunk_record:
                    chunk_text = chunk_record.get("text", "")
                else:
                    # Fallback to searching by index if ID mismatch
                    for c_val in chunks_cache[cache_key].values():
                        if c_val.get("chunk_index") == chunk_index:
                            chunk_text = c_val.get("text", "")
                            break

            # Filter payload system keys to put other properties into metadata
            system_keys = {"chunk_id", "document_id", "chunk_index", "strategy", "page_start", "page_end", "section"}
            metadata = {k: v for k, v in payload.items() if k not in system_keys}

            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk_id or str(point.id),
                document_id=doc_id or "",
                chunk_index=chunk_index if chunk_index is not None else -1,
                score=float(point.score),
                text=chunk_text,
                strategy=strategy,
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                section=payload.get("section"),
                metadata=metadata
            )
            retrieved_chunks.append(retrieved_chunk)

        # 5. Sort highest similarity scores first
        retrieved_chunks.sort(key=lambda x: x.score, reverse=True)

        total_time = time.time() - start_time

        return SearchResponse(
            query=query,
            results=retrieved_chunks,
            search_time_seconds=total_time,
            vector_dimension=len(query_vector),
            retrieved_count=len(retrieved_chunks)
        )
