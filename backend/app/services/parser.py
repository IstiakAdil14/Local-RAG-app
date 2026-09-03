import fitz  # PyMuPDF
import docx
import re
from typing import List, Tuple
from app.schemas.document import Page, DocumentMetadata

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove excessive consecutive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip()

def parse_pdf(file_path: str) -> Tuple[List[Page], DocumentMetadata]:
    pages = []
    metadata = DocumentMetadata()
    
    try:
        doc = fitz.open(file_path)
        
        # Extract metadata
        pdf_meta = doc.metadata
        if pdf_meta:
            metadata.title = pdf_meta.get("title")
            metadata.author = pdf_meta.get("author")
            metadata.creation_date = pdf_meta.get("creationDate")
            
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            normalized_text = normalize_text(text)
            pages.append(Page(page_number=page_num + 1, text=normalized_text))
            
        doc.close()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")
        
    return pages, metadata

def parse_docx(file_path: str) -> Tuple[List[Page], DocumentMetadata]:
    try:
        doc = docx.Document(file_path)
        # DOCX doesn't have a strict concept of pages in python-docx easily.
        # We will treat the entire document as page 1.
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
                
        text = "\n\n".join(full_text)
        normalized_text = normalize_text(text)
        
        # Basic metadata
        core_props = doc.core_properties
        metadata = DocumentMetadata(
            title=core_props.title,
            author=core_props.author,
            creation_date=str(core_props.created) if core_props.created else None
        )
        
        pages = [Page(page_number=1, text=normalized_text)]
        return pages, metadata
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX: {str(e)}")

def parse_txt(file_path: str) -> Tuple[List[Page], DocumentMetadata]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        normalized_text = normalize_text(text)
        pages = [Page(page_number=1, text=normalized_text)]
        return pages, DocumentMetadata()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
            normalized_text = normalize_text(text)
            pages = [Page(page_number=1, text=normalized_text)]
            return pages, DocumentMetadata()
        except Exception as e:
            raise ValueError(f"Failed to parse TXT with fallback encoding: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to parse TXT: {str(e)}")
