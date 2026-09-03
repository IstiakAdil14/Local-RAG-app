import os
import pytest
from app.services.parser import normalize_text, parse_txt

def test_normalize_text():
    # Test excess newlines
    assert normalize_text("Hello\n\n\n\nWorld") == "Hello\n\nWorld"
    
    # Test line endings
    assert normalize_text("Hello\r\nWorld") == "Hello\nWorld"
    
    # Test whitespace stripping
    assert normalize_text("  Hello  ") == "Hello"
    
    # Test English text
    english_text = "This is a test."
    assert normalize_text(f"  {english_text}  ") == english_text

def test_parse_txt(tmp_path):
    # Create temporary txt file
    test_file = tmp_path / "test.txt"
    content = "Hello World\n\nThis is a test document."
    test_file.write_text(content, encoding="utf-8")
    
    pages, metadata = parse_txt(str(test_file))
    
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Hello World" in pages[0].text
    assert "This is a test document." in pages[0].text

def test_parse_txt_invalid_encoding(tmp_path):
    # Test fallback encoding
    test_file = tmp_path / "test_latin.txt"
    test_file.write_bytes(b"Caf\xe9") # Latin-1 encoding
    
    pages, metadata = parse_txt(str(test_file))
    assert len(pages) == 1
    assert "Caf" in pages[0].text
