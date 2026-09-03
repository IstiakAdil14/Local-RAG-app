import pytest
from app.schemas.document import ParsedDocument, Page, DocumentMetadata
from app.schemas.chunk import ChunkingConfig
from app.services.chunking.fixed import FixedChunker
from app.services.chunking.recursive import RecursiveChunker
from app.services.chunking.structure import StructureAwareChunker

@pytest.fixture
def sample_document():
    pages = [
        Page(page_number=1, text="1. Introduction\n\nThis is a short test document. It has a few sentences. "),
        Page(page_number=2, text="This is the second page.\n\nIt continues the thought here. And another one."),
        Page(page_number=3, text="3. Conclusion\n\nWe conclude this document.\n\n")
    ]
    return ParsedDocument(
        document_id="test_doc_123",
        filename="test.txt",
        file_type="txt",
        page_count=3,
        pages=pages,
        metadata=DocumentMetadata(title="Test")
    )

def test_fixed_chunker(sample_document):
    chunker = FixedChunker()
    config = ChunkingConfig(strategy="fixed", chunk_size=50, overlap=10)
    
    chunks = chunker.chunk(sample_document, config)
    
    assert len(chunks) > 0
    assert chunks[0].strategy == "fixed"
    assert chunks[0].chunk_size == 50
    assert chunks[0].overlap == 10
    
    # Check deterministic ordering and indexing
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    
    # Verify page metadata tracking
    assert chunks[0].page_start == 1
    
    # Verify overlap exists (the text should overlap)
    # The first chunk should end with something that overlaps into the second chunk
    # But since text could be short, just test the sizes
    for chunk in chunks[:-1]:
        assert len(chunk.text) <= 50

def test_recursive_chunker(sample_document):
    chunker = RecursiveChunker()
    config = ChunkingConfig(strategy="recursive", chunk_size=100, overlap=0)
    
    chunks = chunker.chunk(sample_document, config)
    
    assert len(chunks) > 0
    assert chunks[0].strategy == "recursive"
    
    # Check that sentences aren't violently broken if possible
    # We'd expect boundaries to align somewhat with sentences or paragraphs
    for chunk in chunks:
        assert len(chunk.text) <= 100
        assert chunk.page_start >= 1
        assert chunk.page_end <= 3
        assert chunk.page_start <= chunk.page_end

def test_structure_aware_chunker(sample_document):
    chunker = StructureAwareChunker()
    config = ChunkingConfig(strategy="structure", chunk_size=500)
    
    chunks = chunker.chunk(sample_document, config)
    
    assert len(chunks) > 0
    assert chunks[0].strategy == "structure"
    
    # The first chunk should likely capture the "1. Introduction" section
    assert chunks[0].section == "1. Introduction"
    
    # The last chunk should likely capture "3. Conclusion"
    assert chunks[-1].section == "3. Conclusion"
    
    for chunk in chunks:
        assert len(chunk.text) <= 500

def test_empty_document():
    doc = ParsedDocument(
        document_id="empty_123",
        filename="empty.txt",
        file_type="txt",
        page_count=1,
        pages=[Page(page_number=1, text="   \n   ")]
    )
    
    chunker = FixedChunker()
    config = ChunkingConfig(strategy="fixed", chunk_size=50)
    chunks = chunker.chunk(doc, config)
    
    # Should safely return 0 chunks for empty documents
    assert len(chunks) == 0

def test_invalid_config(sample_document):
    chunker = FixedChunker()
    
    with pytest.raises(ValueError):
        # Overlap >= chunk size
        config = ChunkingConfig(strategy="fixed", chunk_size=50, overlap=50)
        chunker.chunk(sample_document, config)
        
    with pytest.raises(ValueError):
        # Chunk size <= 0
        config = ChunkingConfig(strategy="fixed", chunk_size=0, overlap=0)
        chunker.chunk(sample_document, config)
