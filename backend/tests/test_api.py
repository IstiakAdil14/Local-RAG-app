import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_invalid_extension():
    # Create a dummy file with unsupported extension
    files = {"file": ("test.csv", b"dummy content", "text/csv")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]

def test_upload_txt_file(tmp_path):
    # Test valid text file
    content = b"This is a test document in English.\n\nHere is some more text."
    files = {"file": ("test.txt", content, "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["file_type"] == "txt"
    assert data["page_count"] == 1
    assert data["status"] == "Success"
    assert "document_id" in data

def test_upload_empty_txt_file():
    # Test empty file
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "Empty file" in response.json()["detail"]
