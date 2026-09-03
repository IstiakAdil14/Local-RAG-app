import os
import json
import time
from typing import List, Dict, Any, Optional
from app.schemas.chunk import DocumentChunk
from app.schemas.embedding import EmbeddingRecord, EmbeddingResult, EmbeddingStatus

class EmbeddingService:
    def __init__(self):
        # We assume project root is one level up from backend/app/services/embeddings/
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        self.processed_dir = os.path.join(self.base_dir, "documents", "processed")
        self.chunks_dir = os.path.join(self.processed_dir, "chunks")
        self.embeddings_dir = os.path.join(self.processed_dir, "embeddings")
        
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.expected_dimension = 384
        # Model instance is stored as a class variable to avoid duplicate loads
        pass

    _shared_model = None

    def _get_model(self):
        """Lazy loads the embedding model."""
        if EmbeddingService._shared_model is None:
            from sentence_transformers import SentenceTransformer
            # We explicitly load the model only once and cache it in memory
            EmbeddingService._shared_model = SentenceTransformer(self.model_name)
        return EmbeddingService._shared_model

    def _load_chunks(self, document_id: str, strategy: str) -> List[DocumentChunk]:
        """Loads chunks for a specific document and strategy."""
        filename = f"{document_id}_chunks_{strategy}.json"
        filepath = os.path.join(self.chunks_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Chunks not found for document {document_id} and strategy {strategy}.")
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [DocumentChunk(**chunk) for chunk in data]

    def get_embedding_status(self, document_id: str, strategy: str) -> EmbeddingStatus:
        """Checks if embeddings exist for a document and strategy."""
        filename = f"{document_id}_embeddings_{strategy}.json"
        filepath = os.path.join(self.embeddings_dir, filename)
        
        if not os.path.exists(filepath):
            return EmbeddingStatus(exists=False, document_id=document_id, strategy=strategy)
            
        # We can read the first few bytes or just the JSON to get stats
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        chunk_count = len(data)
        dimension = data[0].get("embedding_dimension") if chunk_count > 0 else None
        
        return EmbeddingStatus(
            exists=True, 
            document_id=document_id, 
            strategy=strategy, 
            chunk_count=chunk_count, 
            embedding_dimension=dimension,
            metadata={"model": self.model_name}
        )
    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding for a single query string."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
            
        model = self._get_model()
        # encode returns a numpy array, convert to list of floats
        embedding = model.encode(query, show_progress_bar=False)
        return embedding.tolist()

    def generate_embeddings(self, document_id: str, strategy: str, batch_size: int = 16) -> EmbeddingResult:
        """Generates embeddings for chunks of a given document and strategy."""
        chunks = self._load_chunks(document_id, strategy)
        
        if not chunks:
            raise ValueError(f"No chunks found for document {document_id}.")
            
        model = self._get_model()
        
        start_time = time.time()
        
        records: List[EmbeddingRecord] = []
        
        # Batch processing
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            
            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=False)
            
            for j, embedding in enumerate(embeddings):
                # Ensure dimension matches expected
                dim = len(embedding)
                if dim != self.expected_dimension:
                    raise ValueError(f"Dimension mismatch. Expected {self.expected_dimension}, got {dim}.")
                    
                chunk = batch[j]
                
                # We store metadata provenance
                meta = {
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "model": self.model_name
                }
                # Include existing chunk metadata if any
                meta.update(chunk.metadata)
                
                record = EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    embedding_dimension=dim,
                    embedding=embedding.tolist(),
                    metadata=meta
                )
                records.append(record)
                
        processing_time = time.time() - start_time
        
        # Store embeddings locally
        filename = f"{document_id}_embeddings_{strategy}.json"
        filepath = os.path.join(self.embeddings_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([r.dict() for r in records], f, ensure_ascii=False) # Not indenting massively to save space, but could for readability.
            
        return EmbeddingResult(
            document_id=document_id,
            strategy=strategy,
            chunks=len(records),
            embedding_dimension=self.expected_dimension,
            processing_time_seconds=processing_time
        )
