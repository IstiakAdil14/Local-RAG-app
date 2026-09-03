from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section: Optional[str] = None
    strategy: str
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    metadata: Dict[str, Any] = {}

class ChunkingConfig(BaseModel):
    strategy: str
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    
    # Internal validation/defaults can be added here if needed

class ChunkExperimentResult(BaseModel):
    timestamp: str
    document_id: str
    filename: str
    strategy: str
    config: ChunkingConfig
    chunk_count: int
    average_chunk_size: float
    minimum_chunk_size: int
    maximum_chunk_size: int
    median_chunk_size: float
    total_source_characters: int
    total_chunk_characters: int
    coverage: float
    processing_time_seconds: float
    preview_chunks: List[DocumentChunk]
