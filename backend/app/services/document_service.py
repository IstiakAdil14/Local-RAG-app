import hashlib
import os
import json
from fastapi import UploadFile
from app.schemas.document import ParsedDocument, UploadResponse
from app.services.parser import parse_pdf, parse_docx, parse_txt

# Define paths relative to the current file (backend/app/services/document_service.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
RAW_DIR = os.path.join(BASE_DIR, "documents", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "documents", "processed")

def ingest_document(file: UploadFile) -> UploadResponse:
    # Validate extension
    filename = file.filename
    if not filename:
        raise ValueError("Filename missing")
        
    # Sanitize filename strictly to avoid path traversal
    filename = os.path.basename(filename)
        
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = [".pdf", ".docx", ".txt", ".md"]
    if ext not in allowed_exts:
        raise ValueError(f"Unsupported file type. Supported types: {', '.join(allowed_exts)}")
        
    # Read file content
    content = file.file.read()
    if not content:
        raise ValueError("Empty file")
        
    # Generate SHA-256 ID
    document_id = hashlib.sha256(content).hexdigest()
    
    # Save raw file
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, f"{document_id}_{filename}")
    with open(raw_path, "wb") as f:
        f.write(content)
        
    # Parse file
    try:
        if ext == ".pdf":
            pages, metadata = parse_pdf(raw_path)
        elif ext == ".docx":
            pages, metadata = parse_docx(raw_path)
        elif ext in [".txt", ".md"]:
            pages, metadata = parse_txt(raw_path)
        else:
            raise ValueError("Unsupported extension logic error")
    except Exception as e:
        raise ValueError(f"Parsing failed: {str(e)}")
        
    if not pages or all(not page.text.strip() for page in pages):
        if ext == ".pdf":
            raise ValueError("OCR_REQUIRED")
        else:
            raise ValueError("Empty document or unreadable text")
            
    parsed_doc = ParsedDocument(
        document_id=document_id,
        filename=filename,
        file_type=ext[1:],
        page_count=len(pages),
        pages=pages,
        metadata=metadata
    )
    
    # Save processed document
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, f"{document_id}.json")
    with open(processed_path, "w", encoding="utf-8") as f:
        f.write(parsed_doc.model_dump_json(indent=2))
        
    return UploadResponse(
        filename=filename,
        document_id=document_id,
        file_type=ext[1:],
        page_count=len(pages),
        status="Success"
    )
