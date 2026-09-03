from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class VectorDbStatus(BaseModel):
    connected: bool
    collections: List[str] = []
    vector_count: int = 0
    storage_type: str

class DocumentIndexStatus(BaseModel):
    indexed: bool
    vector_count: int = 0
    collection: str = "documents"

class IndexRequest(BaseModel):
    strategy: str = "fixed"

class IndexResponse(BaseModel):
    document_id: str
    vectors_uploaded: int
    collection: str
