from fastapi import APIRouter, HTTPException
from app.schemas.embedding import EmbeddingRequest, EmbeddingResult, EmbeddingStatus
from app.services.embeddings.service import EmbeddingService

router = APIRouter()
embedding_service = EmbeddingService()

@router.get("/{document_id}/embeddings", response_model=EmbeddingStatus)
def get_embedding_status(document_id: str, strategy: str = "fixed"):
    """
    Check if embeddings exist for a given document and strategy.
    """
    try:
        return embedding_service.get_embedding_status(document_id, strategy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/embeddings", response_model=EmbeddingResult)
def generate_embeddings(document_id: str, request: EmbeddingRequest):
    """
    Generate dense vector embeddings for chunks of a given document and strategy.
    """
    try:
        return embedding_service.generate_embeddings(document_id, request.strategy, request.batch_size)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
