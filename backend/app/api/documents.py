from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_service import ingest_document
from app.schemas.document import UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        response = ingest_document(file)
        return response
    except ValueError as e:
        error_msg = str(e)
        if error_msg == "OCR_REQUIRED":
            raise HTTPException(status_code=422, detail="Scanned PDF detected. OCR_REQUIRED")
        elif "Unsupported file type" in error_msg:
            raise HTTPException(status_code=415, detail=error_msg)
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        # In a real app we would log the full exception here
        raise HTTPException(status_code=500, detail="An unexpected error occurred during processing.")
