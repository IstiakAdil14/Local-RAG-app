from fastapi import APIRouter, HTTPException
from app.schemas.vector_store import VectorDbStatus, DocumentIndexStatus, IndexRequest, IndexResponse
from app.services.vector_store.service import QdrantService

router = APIRouter()
vector_service = QdrantService()

@router.get("/status", response_model=VectorDbStatus)
def get_vector_db_status():
    """
    Check Qdrant database connection status and global metrics.
    """
    try:
        return vector_service.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}/index", response_model=DocumentIndexStatus)
def get_document_index_status(document_id: str):
    """
    Check if a specific document's vectors have been indexed into Qdrant.
    """
    try:
        return vector_service.get_document_status(document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/{document_id}/index", response_model=IndexResponse)
def index_document(document_id: str, request: IndexRequest):
    """
    Load embeddings from disk and upsert them into Qdrant collection.
    """
    try:
        return vector_service.index_document(document_id, request.strategy)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
