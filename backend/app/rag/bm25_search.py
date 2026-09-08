import os
import pickle
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.schemas.document import DocumentChunk

class LocalBM25Store:
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self.bm25: BM25Okapi = None
        self.chunks: List[DocumentChunk] = []
        self._load_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _load_index(self):
        if self.index_path and os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]

    def save_index(self):
        if not self.index_path:
            return
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunks": self.chunks
            }, f)

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.chunks.extend(chunks)
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save_index()

    def clear(self):
        self.chunks = []
        self.bm25 = None
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except Exception:
                pass

    def search(self, query: str, top_k: int = 5)-> List[Dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        scored_pairs = sorted(
            enumerate(doc_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]   

        results = []
        for idx, score in scored_pairs:
            chunk = self.chunks[idx]
            results.append({
                "score": float(score),
                "text": chunk.text,
                "metadata": {
                    "document_id": chunk.metadata.document_id,
                    "chunk_id": chunk.metadata.chunk_id,
                    "document_name": chunk.metadata.document_name,
                    "page_number": chunk.metadata.page_number,
                    "section": chunk.metadata.section,
                }
            })
        
        return results
