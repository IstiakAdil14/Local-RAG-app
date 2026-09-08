import os
import pickle
import re
from typing import Dict, List, Optional, Any
from app.schemas.document import DocumentMetadata, DocumentChunk

class DocumentMetadataStore:
    def __init__(self, store_path: str = "./data/doc_metadata.pkl"):
        self.store_path = store_path
        self.store: Dict[str, DocumentMetadata] = {}
        self._load_store()

    def _load_store(self):
        if self.store_path and os.path.exists(self.store_path):
            try:
                with open(self.store_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.store = data
            except Exception as e:
                print(f"⚠️ Failed to load DocumentMetadataStore from {self.store_path}: {e}")
                self.store = {}

    def save_store(self):
        if not self.store_path:
            return
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        try:
            with open(self.store_path, "wb") as f:
                pickle.dump(self.store, f)
        except Exception as e:
            print(f"⚠️ Failed to save DocumentMetadataStore: {e}")

    def add_document(self, metadata: DocumentMetadata):
        self.store[metadata.document_id] = metadata
        self.save_store()

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        return self.store.get(document_id)

    def get_all_documents(self) -> List[DocumentMetadata]:
        return list(self.store.values())

    def get_latest_document(self) -> Optional[DocumentMetadata]:
        if not self.store:
            return None
        return list(self.store.values())[-1]

    def clear(self):
        self.store = {}
        if self.store_path and os.path.exists(self.store_path):
            try:
                os.remove(self.store_path)
            except Exception:
                pass

class DocumentIntelligenceExtractor:
    @staticmethod
    def extract_sections(chunks: List[DocumentChunk]) -> List[str]:
        raw_sections = []
        for c in chunks:
            sec = c.metadata.section
            if sec and sec not in ["General", "General Section"] and len(sec) < 80:
                raw_sections.append(sec.strip())
        seen = set()
        unique_sections = []
        for s in raw_sections:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique_sections.append(s)
        return unique_sections[:15]

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "this", "that", "it", "from", "as", "be"}
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            if w not in stop_words:
                freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:10]]

    @classmethod
    def create_metadata(
        cls,
        doc_id: str,
        doc_name: str,
        title: str,
        pages_data: List[Dict[str, Any]],
        chunks: List[DocumentChunk],
        generator: Optional[Any] = None
    ) -> DocumentMetadata:
        total_pages = len(pages_data) if pages_data else 1
        first_page_text = pages_data[0]["text"] if pages_data else ""
        sections = cls.extract_sections(chunks)
        keywords = cls.extract_keywords(" ".join([p.get("text", "") for p in pages_data[:3]]))

        summary = ""
        if generator and chunks:
            try:
                summary = generator.summarize_document(chunks[:12])
            except Exception as e:
                print(f"⚠️ Index-time summary generation skipped: {e}")

        if not summary:
            if first_page_text:
                summary = f"Document '{doc_name}' contains {total_pages} page(s). First section excerpt: {first_page_text[:300]}..."
            else:
                summary = f"Document '{doc_name}' containing {total_pages} page(s) and {len(chunks)} chunks."

        return DocumentMetadata(
            document_id=doc_id,
            document_name=doc_name,
            title=title,
            total_pages=total_pages,
            first_page_text=first_page_text[:2000],
            sections=sections,
            summary=summary,
            keywords=keywords
        )
