import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import SemanticStructureChunker
from app.rag.embeddings import LocalEmbeddingEngine
from app.rag.vector_search import LocalVectorStore

def test_phase3_pipeline():
    print(">>> 1/4 Parsing and chunking sample text....")
    sample_file = "data/raw/sample_test.txt"
    pages = DocumentParser.parse(sample_file)
    chunker = SemanticStructureChunker(max_chunk_size=120)
    chunks = chunker.chunk(pages, doc_id="DOC_P3", doc_name="sample_test.txt")
    print(f"created {len(chunks)} chunks.")

    print("\n2/4 Initializing BGE-M3 and cumputing emdeddings")
    embedder = LocalEmbeddingEngine()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)
    print(f"✓ Embeddings computed. Count: {len(embeddings)}, Vector dim: {len(embeddings[0])}")

    print("\n3/4 Indexing chunks into local Qdrant")
    vstore = LocalVectorStore(storage_path="./data/qdrant_db",collection_name="test_collection")
    vstore.index_chunks(chunks,embeddings)
    print("Chunks successfully indexed in Qdrant")

    print("\n4/4 Executing vector search query")
    query = "What retrival models are used?"
    q_vec = embedder.embed_query(query)
    results = vstore.search(q_vec,top_k=2)


    assert len(results) > 0, "No results returned from Qdrant vector search."
    print(f"✓ Search returned {len(results)} hit(s).")
    print(f"  Top Match Score: {results[0]['score']:.4f}")
    print(f"  Text: {results[0]['text']}")
    print(f"  Section: {results[0]['metadata']['section']}")

    print("\n✓ Phase 3 embedding and vector storage verified successfully.")

if __name__ == "__main__":
    test_phase3_pipeline()
    
    
     


