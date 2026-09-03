import re
from typing import List, Tuple
from app.schemas.document import ParsedDocument
from app.schemas.chunk import DocumentChunk, ChunkingConfig
from app.services.chunking.base import BaseChunker

class RecursiveChunker(BaseChunker):
    
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
            if len(full_text) > 0 and not full_text.endswith((" ", "\n", "\t")):
                text = " " + text
                
            start = current_idx
            end = current_idx + len(text)
            page_boundaries.append((start, end, page.page_number))
            full_text += text
            current_idx = end
            
        if not full_text:
            return []

        # Separators in order of preference
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def _split_text(text: str, current_sep_idx: int) -> List[str]:
            if len(text) <= chunk_size or current_sep_idx >= len(separators):
                return [text]
                
            sep = separators[current_sep_idx]
            
            if sep == "":
                # Fallback to character splitting
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                
            # Split by separator while keeping the separator
            # Using regex to split and keep separator
            splits = []
            if sep == "\n\n":
                parts = re.split(r'(\n\n)', text)
            elif sep == "\n":
                parts = re.split(r'(\n)', text)
            elif sep == ". ":
                parts = re.split(r'(\. )', text)
            else:
                parts = re.split(r'( )', text)
                
            # Combine parts and separators back together
            combined_parts = []
            current_part = ""
            for i in range(0, len(parts), 2):
                part = parts[i]
                current_sep = parts[i+1] if i+1 < len(parts) else ""
                combined_parts.append(part + current_sep)
                
            # Recursively split any parts that are still too large
            final_parts = []
            for part in combined_parts:
                if len(part) > chunk_size:
                    final_parts.extend(_split_text(part, current_sep_idx + 1))
                elif part:
                    final_parts.append(part)
                    
            return final_parts

        # 1. Split into smallest allowable parts
        raw_splits = _split_text(full_text, 0)
        
        # 2. Combine parts into chunks of size <= chunk_size
        chunks: List[DocumentChunk] = []
        current_chunk_text = ""
        current_chunk_start_idx = 0
        current_global_idx = 0
        chunk_index = 0
        
        i = 0
        while i < len(raw_splits):
            part = raw_splits[i]
            
            if len(current_chunk_text) + len(part) <= chunk_size or not current_chunk_text:
                current_chunk_text += part
                current_global_idx += len(part)
                i += 1
            else:
                # Chunk is full, finalize it
                chunks.append(self._create_chunk_from_text(
                    current_chunk_text, 
                    current_chunk_start_idx, 
                    document.document_id, 
                    chunk_index, 
                    page_boundaries, 
                    chunk_size, 
                    overlap
                ))
                chunk_index += 1
                
                # Handle overlap: take text from the end of current_chunk_text
                if overlap > 0:
                    overlap_text = current_chunk_text[-overlap:]
                    # We need to adjust start idx by how much we dropped
                    dropped_length = len(current_chunk_text) - len(overlap_text)
                    current_chunk_start_idx += dropped_length
                    current_chunk_text = overlap_text
                else:
                    current_chunk_start_idx += len(current_chunk_text)
                    current_chunk_text = ""
                    
        # Add the last remaining chunk
        if current_chunk_text.strip():
            chunks.append(self._create_chunk_from_text(
                current_chunk_text, 
                current_chunk_start_idx, 
                document.document_id, 
                chunk_index, 
                page_boundaries, 
                chunk_size, 
                overlap
            ))

        return chunks

    def _create_chunk_from_text(
        self, text: str, start_idx: int, doc_id: str, chunk_index: int, 
        page_boundaries: List[Tuple[int, int, int]], chunk_size: int, overlap: int
    ) -> DocumentChunk:
        
        end_idx = start_idx + len(text)
        
        page_start = 1
        page_end = 1
        for b_start, b_end, p_num in page_boundaries:
            if start_idx < b_end and b_start <= end_idx:
                if start_idx >= b_start and start_idx < b_end:
                    page_start = p_num
                if end_idx > b_start and end_idx <= b_end:
                    page_end = p_num
                    
        # Ensure page_end is at least page_start (can happen at extreme edges)
        page_end = max(page_start, page_end)
        
        chunk_id = self.generate_chunk_id(doc_id, "recursive", chunk_index, text)
        
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_index=chunk_index,
            text=text.strip(),
            page_start=page_start,
            page_end=page_end,
            strategy="recursive",
            chunk_size=chunk_size,
            overlap=overlap,
            metadata={}
        )
        self.validate_chunk(chunk)
        return chunk
