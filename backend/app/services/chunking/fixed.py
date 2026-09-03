from typing import List, Tuple
from app.schemas.document import ParsedDocument
from app.schemas.chunk import DocumentChunk, ChunkingConfig
from app.services.chunking.base import BaseChunker

class FixedChunker(BaseChunker):
    
    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        chunk_size = config.chunk_size if config.chunk_size is not None else 500
        overlap = config.overlap if config.overlap is not None else 0
        
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
            
        full_text = ""
        page_boundaries: List[Tuple[int, int, int]] = []
        current_idx = 0
        
        for page in document.pages:
            text = page.text
            # Ensure spacing between pages if missing
            if len(full_text) > 0 and not full_text.endswith((" ", "\n", "\t")):
                text = " " + text
                
            start = current_idx
            end = current_idx + len(text)
            page_boundaries.append((start, end, page.page_number))
            full_text += text
            current_idx = end
            
        if not full_text:
            return []

        chunks: List[DocumentChunk] = []
        step = chunk_size - overlap
        index = 0
        
        for chunk_start in range(0, len(full_text), step):
            chunk_end = min(chunk_start + chunk_size, len(full_text))
            chunk_text = full_text[chunk_start:chunk_end]
            
            if not chunk_text.strip():
                continue
                
            # Determine page start and end
            page_start = 1
            page_end = 1
            for b_start, b_end, p_num in page_boundaries:
                if chunk_start < b_end and b_start <= chunk_end:
                    # this page overlaps with our chunk
                    if chunk_start >= b_start and chunk_start < b_end:
                        page_start = p_num
                    if chunk_end > b_start and chunk_end <= b_end:
                        page_end = p_num
                        
            # If the chunk ends exactly on a boundary or goes past the very last character,
            # make sure page_end captures the last valid page
            if chunk_end >= len(full_text) and page_boundaries:
                page_end = page_boundaries[-1][2]
                
            chunk_id = self.generate_chunk_id(document.document_id, "fixed", index, chunk_text)
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                chunk_index=index,
                text=chunk_text,
                page_start=page_start,
                page_end=page_end,
                strategy="fixed",
                chunk_size=chunk_size,
                overlap=overlap,
                metadata={}
            )
            self.validate_chunk(chunk)
            chunks.append(chunk)
            index += 1
            
            # Avoid infinite loop if step is somehow 0 (caught by validation earlier but safety first)
            if step <= 0:
                break
                
            # If we've reached the end of the text, stop
            if chunk_end == len(full_text):
                break
                
        return chunks
