import pytest
import os
import json
import uuid
from app.services.vector_store.service import QdrantService
from qdrant_client import QdrantClient

@pytest.fixture
def temp_vector_dir(tmp_path):
    """Set up temporary directories and mock embeddings for testing vector indexing."""
    base_dir = tmp_path
    processed_dir = base_dir / "documents" / "processed"
    embeddings_dir = processed_dir / "embeddings"
    db_path = processed_dir / "qdrant_db"
    
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(db_path, exist_ok=True)
    
    # Create mock embeddings with dimension 384
    embeddings = [
        {
            "chunk_id": "test_chunk_1",
            "document_id": "test_doc_1",
            "chunk_index": 0,
            "embedding_dimension": 384,
            "embedding": [0.1] * 384,
            "metadata": {
                "page_start": 1,
                "page_end": 1,
                "section": "Intro"
            }
        },
        {
            "chunk_id": "test_chunk_2",
            "document_id": "test_doc_1",
            "chunk_index": 1,
            "embedding_dimension": 384,
            "embedding": [0.2] * 384,
            "metadata": {
                "page_start": 1,
                "page_end": 2,
                "section": "Body"
            }
        }
    ]
    
    emb_path = embeddings_dir / "test_doc_1_embeddings_fixed.json"
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
        
    return base_dir

def test_qdrant_service_lifecycle(temp_vector_dir):
    service = QdrantService()
    
    # Monkeypatch paths
    service.base_dir = temp_vector_dir
    service.processed_dir = os.path.join(temp_vector_dir, "documents", "processed")
    service.embeddings_dir = os.path.join(service.processed_dir, "embeddings")
    service.db_path = os.path.join(service.processed_dir, "qdrant_db")
    
    # 1. Test status (db will be created via local mode)
    status = service.get_status()
    assert status.connected
    assert "documents" in status.collections
    
    # 2. Test document index status initially
    doc_status = service.get_document_status("test_doc_1")
    assert not doc_status.indexed
    assert doc_status.vector_count == 0
    
    # 3. Index document
    index_res = service.index_document("test_doc_1", "fixed")
    assert index_res.document_id == "test_doc_1"
    assert index_res.vectors_uploaded == 2
    assert index_res.collection == "documents"
    
    # 4. Test document index status after indexing
    doc_status = service.get_document_status("test_doc_1")
    assert doc_status.indexed
    assert doc_status.vector_count == 2
    
    # 5. Verify UUIDs are deterministic and upsert is idempotent
    index_res_2 = service.index_document("test_doc_1", "fixed")
    assert index_res_2.vectors_uploaded == 2  # It upserted the exact same 2 points
    
    doc_status_2 = service.get_document_status("test_doc_1")
    assert doc_status_2.vector_count == 2 # Count should remain 2, not 4
    
    
def test_qdrant_dimension_validation(temp_vector_dir):
    service = QdrantService()
    service.base_dir = temp_vector_dir
    service.processed_dir = os.path.join(temp_vector_dir, "documents", "processed")
    service.embeddings_dir = os.path.join(service.processed_dir, "embeddings")
    service.db_path = os.path.join(service.processed_dir, "qdrant_db")
    
    # Corrupt the embedding dimension manually
    emb_path = os.path.join(service.embeddings_dir, "test_doc_2_embeddings_fixed.json")
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "chunk_id": "bad_chunk",
                "document_id": "test_doc_2",
                "chunk_index": 0,
                "embedding_dimension": 128, # Wrong dim
                "embedding": [0.1] * 128,
                "metadata": {}
            }
        ], f)
        
    with pytest.raises(ValueError) as exc:
        service.index_document("test_doc_2", "fixed")
        
    assert "Invalid dimension 128" in str(exc.value)
