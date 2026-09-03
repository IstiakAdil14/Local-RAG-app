import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.schemas.document import ParsedDocument, Page
from app.schemas.chunk import DocumentChunk, ChunkingConfig, ChunkExperimentResult
from app.services.chunking.fixed import FixedChunker
from app.services.chunking.recursive import RecursiveChunker
from app.services.chunking.structure import StructureAwareChunker

class ChunkingService:
    def __init__(self):
        # We assume project root is one level up from backend/app
        # since backend is run from D:\Local-Multilingual-RAG\backend
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.processed_dir = os.path.join(self.base_dir, "documents", "processed")
        self.chunks_dir = os.path.join(self.processed_dir, "chunks")
        self.experiments_dir = os.path.join(self.base_dir, "experiments")
        
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.experiments_dir, exist_ok=True)
        
        self.chunkers = {
            "fixed": FixedChunker(),
            "recursive": RecursiveChunker(),
            "structure": StructureAwareChunker()
        }

    def load_document(self, document_id: str) -> ParsedDocument:
        file_path = os.path.join(self.processed_dir, f"{document_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed document {document_id} not found.")
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ParsedDocument(**data)

    def run_experiment(self, document_id: str, config: ChunkingConfig) -> ChunkExperimentResult:
        doc = self.load_document(document_id)
        
        if config.strategy not in self.chunkers:
            raise ValueError(f"Unknown strategy: {config.strategy}")
            
        chunker = self.chunkers[config.strategy]
        
        start_time = time.time()
        chunks = chunker.chunk(doc, config)
        processing_time = time.time() - start_time
        
        # Calculate statistics
        chunk_count = len(chunks)
        if chunk_count > 0:
            sizes = [len(c.text) for c in chunks]
            avg_size = sum(sizes) / chunk_count
            min_size = min(sizes)
            max_size = max(sizes)
            sizes_sorted = sorted(sizes)
            mid = chunk_count // 2
            if chunk_count % 2 == 0:
                median_size = (sizes_sorted[mid - 1] + sizes_sorted[mid]) / 2.0
            else:
                median_size = sizes_sorted[mid]
            total_chunk_chars = sum(sizes)
        else:
            avg_size = 0.0
            min_size = 0
            max_size = 0
            median_size = 0.0
            total_chunk_chars = 0
            
        total_source_chars = sum(len(p.text) for p in doc.pages)
        coverage = (total_chunk_chars / total_source_chars * 100) if total_source_chars > 0 else 0.0
        
        preview_limit = 5
        
        result = ChunkExperimentResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            document_id=doc.document_id,
            filename=doc.filename,
            strategy=config.strategy,
            config=config,
            chunk_count=chunk_count,
            average_chunk_size=avg_size,
            minimum_chunk_size=min_size,
            maximum_chunk_size=max_size,
            median_chunk_size=median_size,
            total_source_characters=total_source_chars,
            total_chunk_characters=total_chunk_chars,
            coverage=coverage,
            processing_time_seconds=processing_time,
            preview_chunks=chunks[:preview_limit]
        )
        
        # Save chunks
        chunks_filename = f"{document_id}_chunks_{config.strategy}.json"
        chunks_path = os.path.join(self.chunks_dir, chunks_filename)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.dict() for c in chunks], f, indent=2, ensure_ascii=False)
            
        # Save experiment
        experiment_filename = f"{int(time.time())}_{document_id}_{config.strategy}.json"
        experiment_path = os.path.join(self.experiments_dir, experiment_filename)
        with open(experiment_path, "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False)
            
        return result
        
    def list_processed_documents(self) -> List[Dict[str, Any]]:
        docs = []
        if os.path.exists(self.processed_dir):
            for filename in os.listdir(self.processed_dir):
                if filename.endswith(".json") and not filename.startswith("chunks"):
                    path = os.path.join(self.processed_dir, filename)
                    # Simple read to get metadata
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            docs.append({
                                "document_id": data.get("document_id"),
                                "filename": data.get("filename"),
                                "file_type": data.get("file_type"),
                                "page_count": data.get("page_count")
                            })
                    except Exception:
                        pass
        return docs
