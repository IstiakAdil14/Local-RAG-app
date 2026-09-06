import sys
from pathlib import Path

# Add backend directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag.pipeline import AdvancedRAGPipeline
from app.ingestion.service import DocumentIngestionService
from app.schemas.document import QueryResponse

# Global singletons
rag_pipeline: AdvancedRAGPipeline = None
ingestion_service: DocumentIngestionService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, ingestion_service
    print(">>> Starting up Local RAG API services...")
    rag_pipeline = AdvancedRAGPipeline(
        storage_path="./data/qdrant_db",
        collection_name="production_coll",
        bm25_path="./data/production_bm25.pkl"
    )
    ingestion_service = DocumentIngestionService(
        embedder=rag_pipeline.embedder,
        vector_store=rag_pipeline.vector_store,
        bm25_store=rag_pipeline.bm25_store
    )
    yield
    print(">>> Shutting down Local RAG API services...")

app = FastAPI(
    title="Local RAG System API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    retrieval_candidates: int = 10
    top_n_rerank: int = 3

@app.get("/api/v1/health")
def health_check():
    return {"status": "online", "mode": "local_offline"}

@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")
    
    result = await ingestion_service.ingest_file(file)
    return {"message": "Document indexed successfully", "details": result}

@app.post("/api/v1/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return rag_pipeline.query(
        user_query=request.query,
        retrieval_candidates=request.retrieval_candidates,
        top_n_rerank=request.top_n_rerank
    )