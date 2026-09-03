from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Page(BaseModel):
    page_number: int
    text: str

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None

class ParsedDocument(BaseModel):
    document_id: str
    filename: str
    file_type: str
    page_count: int
    pages: List[Page]
    metadata: Optional[DocumentMetadata] = None

class UploadResponse(BaseModel):
    filename: str
    document_id: str
    file_type: str
    page_count: int
    status: str
