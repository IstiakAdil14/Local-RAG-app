import os
import uuid
import tempfile
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
        self.chunker = SemanticStructureChunker(max_chunk_size=1024)
    
    async def ingest_file(self, file: UploadFile) -> Dict[str, Any]:
        file_id = f"DOC_{uuid.uuid4().hex[:8].upper()}"
        ext = os.path.splitext(file.filename)[1].lower()

        # Stream upload into a temporary file that is automatically deleted after processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        try:
            pages = DocumentParser.parse(temp_path)
            chunks = self.chunker.chunk(pages, file_id, file.filename)
            
            if not chunks:
                return {"file_id": file_id, "filename": file.filename, "chunks_indexed": 0}
            
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_texts(texts)
            self.vector_store.index_chunks(chunks, embeddings)
            self.bm25_store.index_chunks(chunks)

            return {
                "file_id": file_id,
                "filename": file.filename,
                "chunks_indexed": len(chunks),
                "pages_parsed": len(pages)
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def clear_database(self) -> Dict[str, str]:
        self.vector_store.clear()
        self.bm25_store.clear()
        return {"message": "All vector and BM25 document indexes cleared successfully."}
               