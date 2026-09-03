from fastapi import APIRouter, HTTPException, Depends
from app.schemas.retrieval import SearchRequest, SearchResponse
from app.services.retrieval.service import RetrievalService

router = APIRouter()
retrieval_service = RetrievalService()

@router.post("/search", response_model=SearchResponse)
def search_retrieval(request: SearchRequest):
    """
    Perform dense vector retrieval on indexed chunks using cosine similarity.
    """
    # Explicit input validations
    query = request.query
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty or whitespace only."
        )

    if request.top_k <= 0 or request.top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20 (inclusive)."
        )

    try:
        response = retrieval_service.retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id
        )
        return response
    except ValueError as ve:
        # e.g., missing collection or invalid parameter values
        if "does not exist" in str(ve) or "not found" in str(ve):
            raise HTTPException(status_code=404, detail=str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        # e.g., database unavailable or connection failures
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        # Catch-all to prevent raw tracebacks from leaking in response
        raise HTTPException(status_code=500, detail=f"Internal retrieval error: {str(e)}")
