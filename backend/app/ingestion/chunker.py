import re
from typing import List, Dict, Any
from app.schemas.document import DocumentChunk, ChunkMetadata

class BaseChunker:
    def chunk(self, pages_data: List[Dict[str, Any]], doc_id: str, doc_name: str) -> List[DocumentChunk]:
        raise NotImplementedError

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
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
    def __init__(self, max_chunk_size: int = 1024, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def _split_into_sentences(self, text: str) -> List[str]:
        return re.split(r'(?<=[.?!])\s+', text)

    def chunk(self, pages_data: List[Dict[str, Any]], doc_id: str, doc_name: str) -> List[DocumentChunk]:
        chunks = []
        chunk_counter = 1

        for page in pages_data:
            page_num = page["page_number"]
            current_section = page.get("section", "General")
            paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]
            
            current_chunk_words = []
            for para in paragraphs:
                if re.match(r'^(#+|\d+\.|\b[A-Z\s]{3,}\b)', para):
                    first_line = para.split("\n")[0]
                    current_section = first_line[:50]

                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    sentence_words = sentence.split()
                    
                    if len(current_chunk_words) + len(sentence_words) > self.max_chunk_size:
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
                            current_chunk_words = []

                    current_chunk_words.extend(sentence_words)

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