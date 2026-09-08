import re
from typing import List, Dict, Any
from app.schemas.document import DocumentChunk, ChunkMetadata

class BaseChunker:
    def chunk(self, pages_data: List[Dict[str, Any]], doc_id: str, doc_name: str) -> List[DocumentChunk]:
        raise NotImplementedError

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 32):
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
    def __init__(self, max_chunk_size: int = 250, min_chunk_size: int = 30):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, pages_data: List[Dict[str, Any]], doc_id: str, doc_name: str) -> List[DocumentChunk]:
        chunks = []
        chunk_counter = 1

        for page in pages_data:
            page_num = page["page_number"]
            current_section = page.get("section", "General")
            
            # Structure-aware splitting: split by double newlines or key-value section headers
            raw_lines = page["text"].split("\n")
            segments = []
            curr_seg = []

            for line in raw_lines:
                clean_line = line.strip()
                if not clean_line:
                    if curr_seg:
                        segments.append("\n".join(curr_seg))
                        curr_seg = []
                    continue

                # Header or field boundary detection (CV / Document key-value pairs)
                if re.match(r'^(#+|\d+\.|\b[A-Z\s]{3,}\b|Name\s*:|Mother\'s\s*Name\s*:|Father\'s\s*Name\s*:|CAREER\s*OBJECTIVE|EDUCATION|PERSONAL\s*INFORMATION)', clean_line, re.IGNORECASE):
                    if curr_seg:
                        segments.append("\n".join(curr_seg))
                        curr_seg = []
                    current_section = clean_line[:50]

                curr_seg.append(clean_line)

            if curr_seg:
                segments.append("\n".join(curr_seg))

            # Assemble segments into optimal-sized chunks
            current_chunk_words = []
            for seg in segments:
                words = seg.split()
                if not words:
                    continue

                if len(current_chunk_words) + len(words) > self.max_chunk_size and current_chunk_words:
                    chunk_text = " ".join(current_chunk_words)
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
                    current_chunk_words = []

                current_chunk_words.extend(words)

            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
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