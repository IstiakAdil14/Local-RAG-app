import os
import requests
import json

API_URL = "http://127.0.0.1:8000"

# Create a test document
test_text = """Abstract
This is a test document for evaluating chunking strategies. It contains multiple paragraphs and sections.

1. Introduction
We evaluate the fixed, recursive, and structure-aware chunking approaches in this experiment. The goal is to see how different chunkers perform on standard English documents. We want to ensure sentences are kept intact where possible.

2. Methodology
We implemented a recursive chunker that splits on paragraphs, then lines, then sentences. This helps maintain semantic boundaries.
"""

with open("test_doc.txt", "w") as f:
    f.write(test_text)

# Upload the document
with open("test_doc.txt", "rb") as f:
    files = {"file": ("test_doc.txt", f, "text/plain")}
    res = requests.post(f"{API_URL}/api/documents/upload", files=files)
    upload_data = res.json()
    doc_id = upload_data["document_id"]
    print("Uploaded document ID:", doc_id)

# Run fixed chunking
fixed_res = requests.post(f"{API_URL}/api/documents/{doc_id}/chunks", json={
    "strategy": "fixed",
    "chunk_size": 100,
    "overlap": 20
})
print("Fixed chunking:", json.dumps(fixed_res.json(), indent=2))

# Run recursive chunking
recursive_res = requests.post(f"{API_URL}/api/documents/{doc_id}/chunks", json={
    "strategy": "recursive",
    "chunk_size": 100,
    "overlap": 0
})
print("Recursive chunking:", json.dumps(recursive_res.json(), indent=2))

# Run structure-aware chunking
structure_res = requests.post(f"{API_URL}/api/documents/{doc_id}/chunks", json={
    "strategy": "structure",
    "chunk_size": 200
})
print("Structure chunking:", json.dumps(structure_res.json(), indent=2))
