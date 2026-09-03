import os
import requests
import json
import time
from app.services.vector_store.service import QdrantService
from app.services.embeddings.service import EmbeddingService
from app.services.chunk_service import ChunkingService
from app.schemas.chunk import ChunkingConfig

# Directly use services to bypass Uvicorn staleness for the experiment report
print("Fetching processed documents...")
API_URL = "http://127.0.0.1:8000"
res = requests.get(f"{API_URL}/api/documents/processed")
docs = res.json()
doc_id = docs[0]["document_id"]
print(f"Using document: {doc_id}")

print("Generating Chunks...")
chunk_svc = ChunkingService()
config = ChunkingConfig(strategy="fixed", chunk_size=1000, chunk_overlap=200)
chunk_svc.run_experiment(doc_id, config)

print("Generating Embeddings via service directly (Phase 4 bypass)...")
emb_svc = EmbeddingService()
emb_svc.generate_embeddings(doc_id, "fixed")
print("Embeddings generated.")

print("\n--- Testing Qdrant Local Integration ---")
qdrant_svc = QdrantService()
status = qdrant_svc.get_status()
print("Qdrant Status:")
print(status)

print(f"\nIndexing document into Qdrant...")
start = time.time()
index_res = qdrant_svc.index_document(doc_id, "fixed")
elapsed = time.time() - start

print("\nINDEXING SUCCESSFUL!")
print(index_res)
print(f"\nPerformance: Uploaded {index_res.vectors_uploaded} vectors in {elapsed:.3f} seconds.")

print("\nVerifying document indexed status in Qdrant...")
doc_status = qdrant_svc.get_document_status(doc_id)
print(doc_status)

print("\nVerifying global DB status again...")
final_status = qdrant_svc.get_status()
print(final_status)
