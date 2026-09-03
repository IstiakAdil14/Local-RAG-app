import uuid
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.schemas.document import DocumentChunk

class LocalVectorStore:
    def __init__(self, storage_path: str = "./data/qdrant_db", collection_name: str = "rag_chunks"):
        os.makedirs(storage_path, exist_ok=True)
        self.client = QdrantClient(path=storage_path)
        self.collection_name = collection_name
        self.vector_dim = 1024  # BGE-M3 dense dimension

        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )

    def index_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata.chunk_id))
            payload = {
                "text": chunk.text,
                "document_id": chunk.metadata.document_id,
                "document_name": chunk.metadata.document_name,
                "page_number": chunk.metadata.page_number,
                "section": chunk.metadata.section,
                "chunk_id": chunk.metadata.chunk_id
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        return [
            {
                "score": hit.score,
                "text": hit.payload["text"],
                "metadata": {
                    "document_id": hit.payload["document_id"],
                    "document_name": hit.payload["document_name"],
                    "page_number": hit.payload["page_number"],
                    "section": hit.payload["section"],
                    "chunk_id": hit.payload["chunk_id"]
                }
            }
            for hit in results.points
        ]