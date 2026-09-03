from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EmbeddingRequest(BaseModel):
    strategy: str = "fixed"
    batch_size: int = 16

class EmbeddingRecord(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    embedding_dimension: int
    embedding: List[float]
    metadata: Dict[str, Any] = {}

class EmbeddingResult(BaseModel):
    document_id: str
    strategy: str
    chunks: int
    embedding_dimension: int
    processing_time_seconds: float

class EmbeddingStatus(BaseModel):
    exists: bool
    document_id: str
    strategy: str
    chunk_count: int = 0
    embedding_dimension: Optional[int] = None
    metadata: Dict[str, Any] = {}
