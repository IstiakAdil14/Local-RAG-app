import re
from typing import List, Dict, Any
from app.schemas.document import DocumentChunk, ChunkMetadata

class BaseChunker:
    def chunk(self, pages_data: List[Dict[str, Any]], doc_id: str, doc_name: str) -> List[DocumentChunk]:
        raise NotImplementedError

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pages_data: List[Dict[str, Any]], doc_id: str, doc_name: str) -> List[DocumentChunk]:
        chunks = []
        chunk_counter = 1

        for page in pages_data:
            text = page["text"]
            page_num = page["page_number"]
            section = page.get("section", "General")
            
            words = text.split()
            if not words:
                continue

            i = 0
            while i < len(words):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                
                chunk_id = f"{doc_id}_{page_num:03d}_{chunk_counter:02d}"
                metadata = ChunkMetadata(
                    document_id=doc_id,
                    document_name=doc_name,
                    page_number=page_num,
                    section=section,
                    chunk_id=chunk_id
                )
                chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
                chunk_counter += 1
                i += (self.chunk_size - self.chunk_overlap)

        return chunks

class SemanticStructureChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 200, min_chunk_size: int = 20):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def is_generic_header(self, line: str) -> bool:
        line_str = line.strip()
        if not line_str or len(line_str) > 70:
            return False
        # Match markdown headers (#), numbered sections (1., 1.1), ALL CAPS titles, or Key-Value headers (Heading:)
        if re.match(r'^(#+|\d+(\.\d+)*\s+|\b[A-Z0-9\s_\-\.]{3,}\b$|^[A-Z][A-Za-z0-9\s]{2,30}:)', line_str):
            return True
        return False

    def chunk(self, pages_data: List[Dict[str, Any]], doc_id: str, doc_name: str) -> List[DocumentChunk]:
        chunks = []
        chunk_counter = 1

        for page in pages_data:
            page_num = page["page_number"]
            raw_lines = page["text"].split("\n")
            
            current_section = page.get("section", "General Section")
            current_words = []

            for line in raw_lines:
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Generic header boundary detection
                if self.is_generic_header(clean_line):
                    if current_words:
                        chunk_text = " ".join(current_words)
                        chunk_id = f"{doc_id}_{page_num:03d}_{chunk_counter:02d}"
                        metadata = ChunkMetadata(
                            document_id=doc_id,
                            document_name=doc_name,
                            page_number=page_num,
                            section=current_section,
                            chunk_id=chunk_id
                        )
                        chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
                        chunk_counter += 1
                        current_words = []

                    current_section = clean_line[:50]

                words = clean_line.split()
                if len(current_words) + len(words) > self.max_chunk_size and current_words:
                    chunk_text = " ".join(current_words)
                    chunk_id = f"{doc_id}_{page_num:03d}_{chunk_counter:02d}"
                    metadata = ChunkMetadata(
                        document_id=doc_id,
                        document_name=doc_name,
                        page_number=page_num,
                        section=current_section,
                        chunk_id=chunk_id
                    )
                    chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
                    chunk_counter += 1
                    current_words = []

                current_words.extend(words)

            if current_words:
                chunk_text = " ".join(current_words)
                chunk_id = f"{doc_id}_{page_num:03d}_{chunk_counter:02d}"
                metadata = ChunkMetadata(
                    document_id=doc_id,
                    document_name=doc_name,
                    page_number=page_num,
                    section=current_section,
                    chunk_id=chunk_id
                )
                chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
                chunk_counter += 1

        return chunks