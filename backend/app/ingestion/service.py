import os
import uuid
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from fastapi import UploadFile

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import SemanticStructureChunker
from app.rag.embeddings import LocalEmbeddingEngine
from app.rag.vector_search import LocalVectorStore
from app.rag.bm25_search import LocalBM25Store
from app.rag.doc_intelligence import DocumentMetadataStore, DocumentIntelligenceExtractor

class DocumentIngestionService:
    def __init__(
        self,
        embedder: LocalEmbeddingEngine,
        vector_store: LocalVectorStore,
        bm25_store: LocalBM25Store,
        doc_metadata_store: Optional[DocumentMetadataStore] = None,
        generator: Optional[Any] = None,
        upload_dir: str = "./data/raw"    
    ):    
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.doc_metadata_store = doc_metadata_store or DocumentMetadataStore()
        self.generator = generator
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
            title = DocumentParser.extract_document_title(temp_path, pages)
            chunks = self.chunker.chunk(pages, file_id, file.filename)
            
            if not chunks:
                return {"file_id": file_id, "filename": file.filename, "chunks_indexed": 0}
            
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_texts(texts)
            self.vector_store.index_chunks(chunks, embeddings)
            self.bm25_store.index_chunks(chunks)

            doc_meta = DocumentIntelligenceExtractor.create_metadata(
                doc_id=file_id,
                doc_name=file.filename,
                title=title,
                pages_data=pages,
                chunks=chunks,
                generator=self.generator
            )
            self.doc_metadata_store.add_document(doc_meta)

            return {
                "file_id": file_id,
                "filename": file.filename,
                "extracted_title": title,
                "chunks_indexed": len(chunks),
                "pages_parsed": len(pages)
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def clear_database(self) -> Dict[str, str]:
        self.vector_store.clear()
        self.bm25_store.clear()
        self.doc_metadata_store.clear()
        return {"message": "All vector, BM25, and metadata document indexes cleared successfully."}

               