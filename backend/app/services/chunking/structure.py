import re
from typing import List
from app.schemas.document import ParsedDocument
from app.schemas.chunk import DocumentChunk, ChunkingConfig
from app.services.chunking.base import BaseChunker

class StructureAwareChunker(BaseChunker):
    
    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        chunk_size = config.chunk_size if config.chunk_size is not None else 1000
        
        chunks: List[DocumentChunk] = []
        chunk_index = 0
        
        current_section = None
        
        for page in document.pages:
            text = page.text
            if not text.strip():
                continue
                
            # Split page into paragraphs
            paragraphs = [p for p in re.split(r'\n\n+', text) if p.strip()]
            
            current_chunk_text = ""
            
            for para in paragraphs:
                para = para.strip()
                
                # Heuristic to detect headings:
                # - Less than 80 chars
                # - Starts with a capital letter or number
                # - Does not end with typical sentence punctuation
                # - No line breaks inside it
                is_heading = (
                    len(para) < 80 
                    and re.match(r'^(?:\d+\.?\s+)?[A-Z]', para)
                    and not para.endswith(('.', '?', '!', ',', ';', ':'))
                    and '\n' not in para
                )
                
                if is_heading:
                    # Flush current chunk if it exists
                    if current_chunk_text.strip():
                        chunk_id = self.generate_chunk_id(document.document_id, "structure", chunk_index, current_chunk_text)
                        chunk = DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document.document_id,
                            chunk_index=chunk_index,
                            text=current_chunk_text.strip(),
                            page_start=page.page_number,
                            page_end=page.page_number,
                            section=current_section,
                            strategy="structure",
                            chunk_size=chunk_size,
                            metadata={}
                        )
                        self.validate_chunk(chunk)
                        chunks.append(chunk)
                        chunk_index += 1
                        current_chunk_text = ""
                        
                    current_section = para
                    current_chunk_text = para
                    continue
                
                # If adding this paragraph exceeds chunk_size, flush
                if current_chunk_text and (len(current_chunk_text) + len(para) + 2) > chunk_size:
                    chunk_id = self.generate_chunk_id(document.document_id, "structure", chunk_index, current_chunk_text)
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        text=current_chunk_text.strip(),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section=current_section,
                        strategy="structure",
                        chunk_size=chunk_size,
                        metadata={}
                    )
                    self.validate_chunk(chunk)
                    chunks.append(chunk)
                    chunk_index += 1
                    # Start a new chunk, potentially with the same section
                    current_chunk_text = para
                else:
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + para
                    else:
                        current_chunk_text = para
                        
                # What if the paragraph itself is larger than chunk_size?
                # For a pure structure chunker, we might tolerate some large chunks,
                # but to avoid unbounded chunks we can hard-split it if it's too large.
                while len(current_chunk_text) > chunk_size:
                    # Break it forcefully, keeping the section
                    split_text = current_chunk_text[:chunk_size]
                    current_chunk_text = current_chunk_text[chunk_size:]
                    
                    chunk_id = self.generate_chunk_id(document.document_id, "structure", chunk_index, split_text)
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        text=split_text.strip(),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section=current_section,
                        strategy="structure",
                        chunk_size=chunk_size,
                        metadata={}
                    )
                    self.validate_chunk(chunk)
                    chunks.append(chunk)
                    chunk_index += 1
            
            # End of page, flush remaining text
            if current_chunk_text.strip():
                chunk_id = self.generate_chunk_id(document.document_id, "structure", chunk_index, current_chunk_text)
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=current_chunk_text.strip(),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section=current_section,
                    strategy="structure",
                    chunk_size=chunk_size,
                    metadata={}
                )
                self.validate_chunk(chunk)
                chunks.append(chunk)
                chunk_index += 1
                current_chunk_text = ""
                
        return chunks
