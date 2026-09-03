from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import documents, chunks, embeddings, vector_db, retrieval

app = FastAPI(
    title="Local RAG API",
    description="Fully Local Document Question Answering",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In a real app, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api")
app.include_router(chunks.router, prefix="/api/documents", tags=["chunks"])
app.include_router(embeddings.router, prefix="/api/documents", tags=["embeddings"])
app.include_router(vector_db.router, prefix="/api/vector-db", tags=["vector-db"])
app.include_router(vector_db.router, prefix="/api", tags=["vector-db"]) # To support /api/documents/{id}/index as requested
app.include_router(retrieval.router, prefix="/api/retrieval", tags=["retrieval"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}
