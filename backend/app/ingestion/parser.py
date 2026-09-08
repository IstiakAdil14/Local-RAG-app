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

    @staticmethod
    def extract_document_title(file_path: str, pages_data: List[Dict[str, Any]]) -> str:
        filename_base = os.path.splitext(os.path.basename(file_path))[0].strip() if file_path else ""
        
        # 1. Try PyMuPDF PDF metadata if available
        if file_path and os.path.splitext(file_path)[1].lower() == ".pdf":
            try:
                doc = fitz.open(file_path)
                meta_title = (doc.metadata.get("title") or "").strip()
                if meta_title and len(meta_title) > 3 and not meta_title.lower().startswith("microsoft word") and not meta_title.lower().endswith(".pdf"):
                    return meta_title
            except Exception:
                pass
                
        # 2. Heuristic extraction from First Page lines
        if pages_data:
            first_page_text = pages_data[0].get("text", "")
            lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

            # 2a. Check for explicit Subject / Course / Title / Topic prefixes
            for idx, line in enumerate(lines[:10]):
                if re.search(r'^(subject|course|title|topic|document)\s*[:\-]', line, re.I):
                    clean_subj = re.sub(r'^(subject|course|title|topic|document)\s*[:\-]\s*', '', line, flags=re.I).strip()
                    if clean_subj and len(clean_subj) > 2:
                        paper_line = ""
                        if idx + 1 < len(lines):
                            next_line = lines[idx + 1].strip()
                            if re.search(r'^(paper|part|sub|module|unit)\s*([ivx0-9]+|one|two|three|four)?\s*[:\-]', next_line, re.I) or "paper" in next_line.lower():
                                paper_line = next_line
                        if paper_line:
                            return f"{clean_subj} {paper_line}"
                        return clean_subj

            # 2b. Filter out institutional headers, exam headers, forms, and instructions
            noise_patterns = [
                r'\b(college|university|school|institute|academy|department|faculty|board)\b',
                r'\b(midterm|examination|final exam|test|assignment|question paper|term exam)\b',
                r'\b(time\s*:|full marks\s*:|marks\s*:|hours|minutes|duration)\b',
                r'\b(instruction|answer script|group\s*[a-z0-9])\b',
                r'^(page\s*\d+|http|www|chapter)'
            ]

            filtered_lines = []
            for line in lines[:8]:
                if len(line) < 4:
                    continue
                if any(re.search(pat, line, re.I) for pat in noise_patterns):
                    continue
                filtered_lines.append(line)

            if filtered_lines:
                # Check if top remaining line is followed by a Paper/Part line
                top_line = filtered_lines[0]
                if len(filtered_lines) > 1 and re.search(r'^(paper|part|sub|module|unit)\s*', filtered_lines[1], re.I):
                    return f"{top_line} {filtered_lines[1]}"
                return top_line

        # 3. Clean Filename fallback
        if filename_base:
            clean_fn = re.sub(r'[_\-]+', ' ', filename_base).strip()
            if clean_fn and len(clean_fn) > 2:
                return clean_fn
        return filename_base or "Untitled Document"

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