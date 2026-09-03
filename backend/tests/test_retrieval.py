import os
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.retrieval.service import RetrievalService
from app.services.embeddings.service import EmbeddingService
from app.services.vector_store.service import QdrantService
from app.schemas.retrieval import SearchRequest, SearchResponse

client = TestClient(app)

@pytest.fixture
def mock_retrieval_env(tmp_path):
    """
    Sets up a complete mock environment with:
    - Raw chunk files
    - Embedding files
    - Qdrant database collection containing indexed chunks of doc_1 and doc_2
    """
    base_dir = tmp_path
    processed_dir = base_dir / "documents" / "processed"
    chunks_dir = processed_dir / "chunks"
    embeddings_dir = processed_dir / "embeddings"
    db_path = processed_dir / "qdrant_db"
    
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(db_path, exist_ok=True)
    
    # 1. Create mock chunks on disk
    doc1_chunks = [
        {
            "chunk_id": "doc1_c1",
            "document_id": "doc_1",
            "chunk_index": 0,
            "text": "Introduction to Machine Learning. ML is a subset of AI.",
            "page_start": 1,
            "page_end": 1,
            "section": "Introduction",
            "strategy": "fixed",
            "metadata": {}
        },
        {
            "chunk_id": "doc1_c2",
            "document_id": "doc_1",
            "chunk_index": 1,
            "text": "Deep Learning models use multi-layer artificial neural networks.",
            "page_start": 2,
            "page_end": 3,
            "section": "Deep Learning",
            "strategy": "fixed",
            "metadata": {}
        }
    ]
    doc2_chunks = [
        {
            "chunk_id": "doc2_c1",
            "document_id": "doc_2",
            "chunk_index": 0,
            "text": "Python is a popular programming language for data science and web development.",
            "page_start": 1,
            "page_end": 1,
            "section": "Overview",
            "strategy": "fixed",
            "metadata": {}
        }
    ]
    
    with open(chunks_dir / "doc_1_chunks_fixed.json", "w", encoding="utf-8") as f:
        json.dump(doc1_chunks, f)
    with open(chunks_dir / "doc_2_chunks_fixed.json", "w", encoding="utf-8") as f:
        json.dump(doc2_chunks, f)
        
    # 2. Create mock embeddings (384 dimensions)
    # We will make doc1_c1 highly match ML queries, doc1_c2 medium match, doc2_c1 low/zero match.
    # Note: cosine similarity is based on dot product of normalized vectors.
    emb_doc1 = [
        {
            "chunk_id": "doc1_c1",
            "document_id": "doc_1",
            "chunk_index": 0,
            "embedding_dimension": 384,
            "embedding": [0.05] * 384, # Closer to positive
            "metadata": {
                "page_start": 1,
                "page_end": 1,
                "section": "Introduction"
            }
        },
        {
            "chunk_id": "doc1_c2",
            "document_id": "doc_1",
            "chunk_index": 1,
            "embedding_dimension": 384,
            "embedding": [0.01] * 384,
            "metadata": {
                "page_start": 2,
                "page_end": 3,
                "section": "Deep Learning"
            }
        }
    ]
    emb_doc2 = [
        {
            "chunk_id": "doc2_c1",
            "document_id": "doc_2",
            "chunk_index": 0,
            "embedding_dimension": 384,
            "embedding": [-0.05] * 384, # Negative similarity
            "metadata": {
                "page_start": 1,
                "page_end": 1,
                "section": "Overview"
            }
        }
    ]
    
    with open(embeddings_dir / "doc_1_embeddings_fixed.json", "w", encoding="utf-8") as f:
        json.dump(emb_doc1, f)
    with open(embeddings_dir / "doc_2_embeddings_fixed.json", "w", encoding="utf-8") as f:
        json.dump(emb_doc2, f)
        
    # Initialize services pointing to this temp folder
    qdrant_svc = QdrantService()
    qdrant_svc.base_dir = str(base_dir)
    qdrant_svc.processed_dir = str(processed_dir)
    qdrant_svc.embeddings_dir = str(embeddings_dir)
    qdrant_svc.db_path = str(db_path)
    
    # Clean/create collection
    qdrant_svc._ensure_collection()
    
    # Index both docs
    qdrant_svc.index_document("doc_1", "fixed")
    qdrant_svc.index_document("doc_2", "fixed")
    
    # Initialize RetrievalService with the patched qdrant service
    retrieval_svc = RetrievalService(qdrant_service=qdrant_svc)
    retrieval_svc.chunks_dir = str(chunks_dir)
    
    return retrieval_svc


def test_query_embedding_generation(mock_retrieval_env):
    """Verify that query embedding can be generated with correct dimensions."""
    svc = mock_retrieval_env
    query = "What is artificial intelligence?"
    embedding = svc.embedding_service.embed_query(query)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(val, float) for val in embedding)


def test_empty_query_rejection(mock_retrieval_env):
    """Verify that retrieval raises ValueError on empty or whitespace queries."""
    svc = mock_retrieval_env
    
    with pytest.raises(ValueError) as exc:
        svc.retrieve_chunks("")
    assert "Query cannot be empty" in str(exc.value)
    
    with pytest.raises(ValueError) as exc:
        svc.retrieve_chunks("   ")
    assert "Query cannot be empty" in str(exc.value)


def test_top_k_bounds_validation(mock_retrieval_env):
    """Verify that top_k parameters are bounded (gt 0, le 20)."""
    svc = mock_retrieval_env
    
    with pytest.raises(ValueError) as exc:
        svc.retrieve_chunks("Machine learning", top_k=0)
    assert "top_k must be greater than 0" in str(exc.value)
    
    with pytest.raises(ValueError) as exc:
        svc.retrieve_chunks("Machine learning", top_k=-5)
    assert "top_k must be greater than 0" in str(exc.value)
    
    with pytest.raises(ValueError) as exc:
        svc.retrieve_chunks("Machine learning", top_k=21)
    assert "top_k cannot exceed 20" in str(exc.value)


def test_retrieval_ordering(mock_retrieval_env):
    """Verify that returned results are sorted by similarity score, highest first."""
    svc = mock_retrieval_env
    
    # Run retrieval
    res = svc.retrieve_chunks("machine learning", top_k=3)
    
    assert len(res.results) >= 2
    scores = [r.score for r in res.results]
    
    # Check that scores are descending
    assert scores == sorted(scores, reverse=True)


def test_document_filtering(mock_retrieval_env):
    """Verify that supplying document_id limits the search only to chunks of that document."""
    svc = mock_retrieval_env
    
    # Search all documents
    res_all = svc.retrieve_chunks("programming", top_k=5)
    doc_ids_all = {r.document_id for r in res_all.results}
    assert "doc_1" in doc_ids_all
    assert "doc_2" in doc_ids_all
    
    # Search doc_2 only
    res_filtered = svc.retrieve_chunks("programming", top_k=5, document_id="doc_2")
    for r in res_filtered.results:
        assert r.document_id == "doc_2"
        
    # Search doc_1 only
    res_filtered_1 = svc.retrieve_chunks("programming", top_k=5, document_id="doc_1")
    for r in res_filtered_1.results:
        assert r.document_id == "doc_1"


def test_api_endpoint_success(mock_retrieval_env, monkeypatch):
    """Test standard success case of the retrieval search API endpoint."""
    # Monkeypatch the global retrieval service in api.retrieval
    from app.api.retrieval import retrieval_service
    monkeypatch.setattr("app.api.retrieval.retrieval_service", mock_retrieval_env)
    
    response = client.post("/api/retrieval/search", json={
        "query": "artificial intelligence",
        "top_k": 2
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "artificial intelligence"
    assert "results" in data
    assert len(data["results"]) <= 2
    
    # Verify properties
    first_result = data["results"][0]
    assert "chunk_id" in first_result
    assert "document_id" in first_result
    assert "score" in first_result
    assert "text" in first_result
    assert first_result["text"] != "" # Verify text was loaded successfully from mock chunk file


def test_api_endpoint_validations(mock_retrieval_env, monkeypatch):
    """Test validation errors in the API search endpoint."""
    from app.api.retrieval import retrieval_service
    monkeypatch.setattr("app.api.retrieval.retrieval_service", mock_retrieval_env)
    
    # 1. Empty query
    response = client.post("/api/retrieval/search", json={
        "query": "",
        "top_k": 5
    })
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]
    
    # 2. Blank whitespace query
    response = client.post("/api/retrieval/search", json={
        "query": "    ",
        "top_k": 5
    })
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]
    
    # 3. Invalid top_k <= 0
    response = client.post("/api/retrieval/search", json={
        "query": "Valid query",
        "top_k": 0
    })
    assert response.status_code == 400
    assert "top_k must be" in response.json()["detail"]
    
    # 4. Invalid top_k > 20
    response = client.post("/api/retrieval/search", json={
        "query": "Valid query",
        "top_k": 21
    })
    assert response.status_code == 400
    assert "top_k must be" in response.json()["detail"]


def test_missing_collection_handling(mock_retrieval_env, monkeypatch):
    """Verify that search handles missing collections gracefully."""
    svc = mock_retrieval_env
    # Change collection name to something non-existent
    monkeypatch.setattr(svc.qdrant_service, "collection_name", "non_existent_collection")
    
    with pytest.raises(ValueError) as exc:
        svc.retrieve_chunks("test query")
    assert "does not exist" in str(exc.value)
