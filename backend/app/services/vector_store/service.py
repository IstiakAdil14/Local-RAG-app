import os
import json
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.schemas.vector_store import VectorDbStatus, DocumentIndexStatus, IndexResponse

class QdrantService:
    def __init__(self):
        # Determine paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        self.processed_dir = os.path.join(self.base_dir, "documents", "processed")
        self.embeddings_dir = os.path.join(self.processed_dir, "embeddings")
        self.db_path = os.path.join(self.processed_dir, "qdrant_db")
        
        os.makedirs(self.db_path, exist_ok=True)
        
        self.collection_name = "documents"
        self.expected_dimension = 384
        self._client = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            # We use Qdrant local mode (SQLite/RocksDB) because Docker is not available on the machine
            self._client = QdrantClient(path=self.db_path)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        """Ensures that the required collection exists in Qdrant."""
        client = self._client
        try:
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.expected_dimension, 
                        distance=models.Distance.COSINE
                    ),
                )
        except Exception as e:
            print(f"Error checking/creating collection: {e}")

    def get_status(self) -> VectorDbStatus:
        try:
            client = self._get_client()
            collections = client.get_collections().collections
            col_names = [c.name for c in collections]
            
            total_vectors = 0
            if self.collection_name in col_names:
                info = client.get_collection(self.collection_name)
                total_vectors = info.points_count or 0
                
            return VectorDbStatus(
                connected=True,
                collections=col_names,
                vector_count=total_vectors,
                storage_type="local_path"
            )
        except Exception as e:
            return VectorDbStatus(connected=False, storage_type="local_path")

    def get_document_status(self, document_id: str) -> DocumentIndexStatus:
        try:
            client = self._get_client()
            
            # Use filter to count vectors for this specific document
            count_result = client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
            
            count = count_result.count
            return DocumentIndexStatus(
                indexed=(count > 0),
                vector_count=count,
                collection=self.collection_name
            )
        except Exception:
            return DocumentIndexStatus(indexed=False)

    def _generate_uuid(self, chunk_id: str) -> str:
        """Deterministically convert a chunk_id into a standard UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def index_document(self, document_id: str, strategy: str) -> IndexResponse:
        """Loads embeddings from disk and upserts them into Qdrant."""
        filename = f"{document_id}_embeddings_{strategy}.json"
        filepath = os.path.join(self.embeddings_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embeddings not found for {document_id}")
            
        with open(filepath, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        if not records:
            raise ValueError("Embedding file is empty.")
            
        client = self._get_client()
        points = []
        
        for record in records:
            dim = record.get("embedding_dimension")
            if dim != self.expected_dimension:
                raise ValueError(f"Invalid dimension {dim}. Expected {self.expected_dimension}.")
                
            chunk_id = record["chunk_id"]
            point_id = self._generate_uuid(chunk_id)
            
            # Construct metadata payload safely
            payload = {
                "chunk_id": chunk_id,
                "document_id": record["document_id"],
                "chunk_index": record["chunk_index"],
                "strategy": strategy,
                # Explicitly pull metadata keys for standard tracking
                "page_start": record.get("metadata", {}).get("page_start"),
                "page_end": record.get("metadata", {}).get("page_end"),
                "section": record.get("metadata", {}).get("section")
            }
            # Append any remaining raw metadata
            payload.update(record.get("metadata", {}))
            
            point = models.PointStruct(
                id=point_id,
                vector=record["embedding"],
                payload=payload
            )
            points.append(point)
            
        # Bulk upsert (we can do batches if array is huge, but usually fine for a few thousands)
        # Qdrant client handles it internally if we pass a list
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            total_upserted += len(batch)
            
        return IndexResponse(
            document_id=document_id,
            vectors_uploaded=total_upserted,
            collection=self.collection_name
        )
