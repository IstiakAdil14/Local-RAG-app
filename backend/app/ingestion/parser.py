import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image
import os
import io
import re
from typing import List, Dict, Any

def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    # Strip raw PDF header/footer noise (e.g. Page1of2, Page 2 of 2)
    cleaned = re.sub(r'Page\s*\d+\s*(of|\/)\s*\d+', '', text, flags=re.IGNORECASE)
    # Normalize multiple line breaks and spaces
    cleaned = re.sub(r'\r\n|\r', '\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    return cleaned.strip()

class DocumentParser:
    @staticmethod
    def extract_from_txt(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = clean_extracted_text(f.read())
        return [{"page_number": 1, "text": text, "section": "General"}]

    @staticmethod
    def extract_from_docx(file_path: str) -> List[Dict[str, Any]]:
        doc = docx.Document(file_path)
        full_text = []
        current_section = "General"     
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            if para.style.name.startswith("Heading"):
                current_section = text
            full_text.append(f"[{current_section}] : {text}")  
        clean_text = clean_extracted_text("\n".join(full_text))
        return [{"page_number": 1, "text": clean_text, "section": "General"}]

    @staticmethod
    def extract_from_pdf(file_path: str, min_char_threshold: int = 40) -> List[Dict[str, Any]]:
        doc = fitz.open(file_path)
        extracted_pages = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text().strip()

            if len(text) < min_char_threshold:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img).strip()
                except Exception as e:
                    print(f"OCR failed for page {page_num}: {e}")

            clean_text = clean_extracted_text(text)
            lines = [l.strip() for l in clean_text.split("\n") if l.strip()]        
            section = lines[0] if lines else "General"

            extracted_pages.append({
                "page_number": page_num,
                "text": clean_text,
                "section": section
            })

        return extracted_pages

    @classmethod
    def parse(cls, file_path: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return cls.extract_from_pdf(file_path)
        elif ext == ".docx":
            return cls.extract_from_docx(file_path)
        elif ext in [".txt", ".md"]:
            return cls.extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")