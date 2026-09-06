import os
import uuid
import shutil
from typing import Dict, List, Any
from fastapi import UploadFile

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import SemanticStructureChunker
from app.rag.embeddings import LocalEmbeddingEngine
from app.rag.vector_search import LocalVectorStore
from app.rag.bm25_search import LocalBM25Store

class DocumentIngestionService:
    def __init__(
        self,
        embedder: LocalEmbeddingEngine,
        vector_store: LocalVectorStore,
        bm25_store: LocalBM25Store,
        upload_dir: str = "./data/raw"    
    ):    
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.upload_dir = upload_dir
        self.chunker = SemanticStructureChunker(max_chunk_size=512)
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def ingest_file(self, file: UploadFile)-> Dict[str, Any]:
        file_id = f"DOC_{uuid.uuid4().hex[:8].upper()}"
        file_path = os.path.join(self.upload_dir, f"{file_id}_{file.filename}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pages = DocumentParser.parse(file_path)

        chunks = self.chunker.chunk(pages, file_id, file.filename)
        
        if not chunks:
            return {"file_id": file_id, "filename": file.filename, "chunks_indexed": 0}
        
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.index_chunks(chunks, embeddings)
        self.bm25_store.index_chunks(chunks)

        return{
            "file_id": file_id,
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "pages_parsed": len(pages)
        } 
               