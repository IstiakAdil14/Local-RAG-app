from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.schemas.chunk import ChunkingConfig, ChunkExperimentResult
from app.services.chunk_service import ChunkingService

router = APIRouter()
chunk_service = ChunkingService()

@router.get("/processed", response_model=List[Dict[str, Any]])
def list_processed_documents():
    """
    List all documents that have been successfully parsed and are available for chunking.
    """
    try:
        return chunk_service.list_processed_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/chunks", response_model=ChunkExperimentResult)
def run_chunking_experiment(document_id: str, config: ChunkingConfig):
    """
    Run a chunking strategy on a parsed document and return experiment statistics.
    """
    try:
        result = chunk_service.run_experiment(document_id, config)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
