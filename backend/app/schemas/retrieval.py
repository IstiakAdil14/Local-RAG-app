from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: Optional[str] = None

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    score: float
    text: str = "" # We might not have text immediately depending on Qdrant payload, but we'll try to extract it from payload or chunk files
    strategy: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]
    search_time_seconds: float
    vector_dimension: int
    retrieved_count: int
