import os 
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.rag.pipeline import AdvancedRAGPipeline
from app.ingestion.chunker import SemanticStructureChunker
from app.schemas.document import DocumentChunk, ChunkMetadata

def test_intent_detection():
    print(">>> Testing Global Aggregation Query Intent Classifier...")
    pipeline = AdvancedRAGPipeline.__new__(AdvancedRAGPipeline)
    
    global_queries = [
        "give me a list about every cell",
        "list all cells",
        "summarize document",
        "give an overview of all sections",
        "list every cell",
        "explain all cells"
    ]
    
    targeted_queries = [
        "What evaluation metrics are computed in Cell 15?",
        "How does Cell 7 handle price outliers?",
        "Which columns are dropped in Cell 8?"
    ]
    
    for q in global_queries:
        assert pipeline._is_global_query(q) == True, f"Failed to detect global query: {q}"
        print(f"[OK] Detected global query: '{q}'")
        
    for q in targeted_queries:
        assert pipeline._is_global_query(q) == False, f"Erroneously flagged targeted query as global: {q}"
        print(f"[OK] Detected targeted query: '{q}'")

def test_chunker_defaults():
    print("\n>>> Testing SemanticStructureChunker configuration...")
    chunker = SemanticStructureChunker()
    assert chunker.max_chunk_size == 1024, f"Expected 1024 max_chunk_size, got {chunker.max_chunk_size}"
    print(f"[OK] SemanticStructureChunker default max_chunk_size = {chunker.max_chunk_size}")

if __name__ == "__main__":
    test_intent_detection()
    test_chunker_defaults()
    print("\n[OK] Unit verification tests passed successfully!")
