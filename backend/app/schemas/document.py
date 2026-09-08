from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChunkMetadata(BaseModel):
    document_id: str
    document_name: str
    page_number: int
    section: str = "General"
    chunk_id: str

class DocumentChunk(BaseModel):
    text: str
    metadata: ChunkMetadata

class DocumentMetadata(BaseModel):
    document_id: str
    document_name: str
    title: str = ""
    total_pages: int = 1
    first_page_text: str = ""
    sections: List[str] = []
    summary: str = ""
    keywords: List[str] = []

class ProcessedDocument(BaseModel):
    document_id: str
    document_name: str
    total_pages: int
    chunks: List[DocumentChunk]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    retrival_strategy: str = "hybrid"
    chunks_strategy: str = "semantic"

class Citation(BaseModel):
    document_name: str
    page_number: int
    section: str
    chunk_id: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    query_intent: Optional[str] = None
    retrieval_strategy_used: Optional[str] = None