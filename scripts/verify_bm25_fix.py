import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.rag.bm25_search import LocalBM25Store
from app.schemas.document import DocumentChunk, ChunkMetadata

def test_bm25_multi_document_indexing():
    print(">>> Testing LocalBM25Store Multi-Document Indexing Alignment...")
    
    test_pkl = "./data/test_bm25_temp.pkl"
    if os.path.exists(test_pkl):
        os.remove(test_pkl)

    bm25_store = LocalBM25Store(index_path=test_pkl)

    doc1_chunks = [
        DocumentChunk(
            text=f"Section {i} evaluates customer churn metrics and revenue projections",
            metadata=ChunkMetadata(document_id="DOC1", document_name="doc1.txt", page_number=i, section=f"Sec{i}", chunk_id=f"DOC1_{i:02d}")
        ) for i in range(1, 4)
    ]
    bm25_store.index_chunks(doc1_chunks)

    doc2_chunks = [
        DocumentChunk(
            text=f"Section {i} describes general data workflow and features",
            metadata=ChunkMetadata(document_id="DOC2", document_name="doc2.txt", page_number=i, section=f"Sec{i}", chunk_id=f"DOC2_{i:02d}")
        ) for i in range(1, 4)
    ]
    doc2_chunks[0].text = "Section 1 describes unique xgboost_housing_model architecture and features"
    bm25_store.index_chunks(doc2_chunks)

    results = bm25_store.search("xgboost_housing_model", top_k=2)
    print("Search results:", [(round(r['score'], 4), r['metadata']['document_name'], r['metadata']['chunk_id']) for r in results])
    
    top_doc = results[0]["metadata"]["document_name"]
    print(f"Query returned top document: {top_doc}")
    assert top_doc == "doc2.txt", f"Expected doc2.txt, but got {top_doc}!"

    bm25_store.clear()
    print("[OK] BM25 Index alignment test passed cleanly!")

if __name__ == "__main__":
    test_bm25_multi_document_indexing()
