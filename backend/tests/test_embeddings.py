import pytest
import os
import json
from app.schemas.chunk import DocumentChunk
from app.services.embeddings.service import EmbeddingService

@pytest.fixture
def temp_chunks_dir(tmp_path):
    """Set up a temporary chunks directory with mock chunks."""
    base_dir = tmp_path
    processed_dir = base_dir / "documents" / "processed"
    chunks_dir = processed_dir / "chunks"
    embeddings_dir = processed_dir / "embeddings"
    
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Create mock chunks
    chunks = [
        DocumentChunk(
            chunk_id="test_chunk_1",
            document_id="test_doc_1",
            chunk_index=0,
            text="This is the first test sentence for embeddings.",
            page_start=1,
            page_end=1,
            strategy="fixed",
            chunk_size=100
        ),
        DocumentChunk(
            chunk_id="test_chunk_2",
            document_id="test_doc_1",
            chunk_index=1,
            text="Here is another sentence that needs to be vectorized.",
            page_start=1,
            page_end=1,
            strategy="fixed",
            chunk_size=100
        )
    ]
    
    chunks_path = chunks_dir / "test_doc_1_chunks_fixed.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([c.dict() for c in chunks], f)
        
    return base_dir

def test_embedding_service(temp_chunks_dir, monkeypatch):
    """Test the complete embedding service lifecycle."""
    service = EmbeddingService()
    
    # Monkeypatch the base_dir to our temporary test directory
    service.base_dir = temp_chunks_dir
    service.processed_dir = os.path.join(temp_chunks_dir, "documents", "processed")
    service.chunks_dir = os.path.join(service.processed_dir, "chunks")
    service.embeddings_dir = os.path.join(service.processed_dir, "embeddings")
    
    # 1. Test status when no embeddings exist
    status = service.get_embedding_status("test_doc_1", "fixed")
    assert not status.exists
    
    # 2. Generate embeddings
    result = service.generate_embeddings("test_doc_1", "fixed", batch_size=2)
    
    assert result.document_id == "test_doc_1"
    assert result.strategy == "fixed"
    assert result.chunks == 2
    assert result.embedding_dimension == 384
    assert result.processing_time_seconds > 0
    
    # 3. Test status when embeddings exist
    status = service.get_embedding_status("test_doc_1", "fixed")
    assert status.exists
    assert status.chunk_count == 2
    assert status.embedding_dimension == 384
    
    # 4. Validate output file structure
    embeddings_file = os.path.join(service.embeddings_dir, "test_doc_1_embeddings_fixed.json")
    assert os.path.exists(embeddings_file)
    
    with open(embeddings_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["chunk_id"] == "test_chunk_1"
        assert len(data[0]["embedding"]) == 384
        assert isinstance(data[0]["embedding"][0], float)
        assert "page_start" in data[0]["metadata"]
        assert data[0]["metadata"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        
def test_missing_chunks(temp_chunks_dir, monkeypatch):
    """Test generating embeddings for non-existent chunks."""
    service = EmbeddingService()
    service.base_dir = temp_chunks_dir
    service.processed_dir = os.path.join(temp_chunks_dir, "documents", "processed")
    service.chunks_dir = os.path.join(service.processed_dir, "chunks")
    service.embeddings_dir = os.path.join(service.processed_dir, "embeddings")
    
    with pytest.raises(FileNotFoundError):
        service.generate_embeddings("nonexistent_doc", "fixed")
