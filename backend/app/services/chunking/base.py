from abc import ABC, abstractmethod
import hashlib
from typing import List, Optional
from app.schemas.document import ParsedDocument
from app.schemas.chunk import DocumentChunk, ChunkingConfig

class BaseChunker(ABC):
    
    @abstractmethod
    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """
        Takes a ParsedDocument and a ChunkingConfig, returns a list of DocumentChunks.
        """
        pass
        
    def generate_chunk_id(self, document_id: str, strategy: str, chunk_index: int, text: str) -> str:
        """
        Generates a deterministic SHA-256 chunk ID based on doc ID, strategy, index, and text content.
        """
        content = f"{document_id}_{strategy}_{chunk_index}_{text}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()
        
    def validate_chunk(self, chunk: DocumentChunk) -> None:
        """
        Validates that a generated chunk meets minimum requirements.
        Raises ValueError if invalid.
        """
        if not chunk.chunk_id:
            raise ValueError("Chunk must have a chunk_id")
        if not chunk.document_id:
            raise ValueError("Chunk must have a document_id")
        if chunk.chunk_index < 0:
            raise ValueError("Chunk index must be >= 0")
        if not chunk.text or len(chunk.text.strip()) == 0:
            raise ValueError("Chunk text cannot be empty")
        if chunk.page_start < 1 or chunk.page_end < 1:
            raise ValueError("Chunk page numbers must be >= 1")
        if chunk.page_start > chunk.page_end:
            raise ValueError(f"chunk page_start ({chunk.page_start}) cannot be > page_end ({chunk.page_end})")
