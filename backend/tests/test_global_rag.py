import os
import pytest
import shutil
import tempfile
from app.rag.query_classifier import QueryClassifier, QueryIntent
from app.rag.doc_intelligence import DocumentMetadataStore, DocumentIntelligenceExtractor
from app.schemas.document import DocumentMetadata, DocumentChunk, ChunkMetadata
from app.ingestion.parser import DocumentParser
from app.rag.pipeline import AdvancedRAGPipeline

def test_query_classifier():
    assert QueryClassifier.classify("What's the title of this PDF?") == QueryIntent.METADATA
    assert QueryClassifier.classify("What is the title of the document?") == QueryIntent.METADATA
    assert QueryClassifier.classify("How many pages are in this document?") == QueryIntent.METADATA
    assert QueryClassifier.classify("What is the filename?") == QueryIntent.METADATA

    assert QueryClassifier.classify("Summarize this document") == QueryIntent.SUMMARY
    assert QueryClassifier.classify("What is this document about?") == QueryIntent.SUMMARY
    assert QueryClassifier.classify("Give me an overview of the file") == QueryIntent.SUMMARY

    assert QueryClassifier.classify("List all sections in the document") == QueryIntent.LIST
    assert QueryClassifier.classify("What is eclampsia?") == QueryIntent.FACT
    assert QueryClassifier.classify("What is the candidate's father's name?") == QueryIntent.FACT

def test_document_metadata_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test_metadata.pkl")
        store = DocumentMetadataStore(store_path=store_path)

        meta = DocumentMetadata(
            document_id="DOC_TEST1",
            document_name="B413_Midwifery.pdf",
            title="B413 Midwifery, P-II (Gynecological & Obstetrical Nursing)",
            total_pages=45,
            first_page_text="B413 Midwifery, P-II (Gynecological & Obstetrical Nursing)\nPage 1 of 45\nIntroduction to Midwifery Nursing...",
            sections=["Introduction", "Gynecological Nursing"],
            summary="Course guide for B413 Midwifery."
        )

        store.add_document(meta)

        retrieved = store.get_document("DOC_TEST1")
        assert retrieved is not None
        assert retrieved.title == "B413 Midwifery, P-II (Gynecological & Obstetrical Nursing)"
        assert retrieved.total_pages == 45

        # Test persistent reload
        store2 = DocumentMetadataStore(store_path=store_path)
        latest = store2.get_latest_document()
        assert latest is not None
        assert latest.document_id == "DOC_TEST1"

def test_parser_title_extraction():
    pages = [
        {"page_number": 1, "text": "B413 Midwifery, P-II (Gynecological & Obstetrical Nursing)\nPage 1 of 45\nCourse Outline"}
    ]
    title = DocumentParser.extract_document_title("sample_course.pdf", pages)
    assert title == "B413 Midwifery, P-II (Gynecological & Obstetrical Nursing)"

def test_parser_title_extraction_with_exam_header():
    exam_text = (
        "North East Nursing College, Sylhet\n"
        "4th Year B.Sc. in Nursing Midterm Examination, July-2025\n"
        "Subject: B-431 Midwifery & Obstetrical Nursing\n"
        "Paper II: Gynecological and Obstetrical Nursing\n"
        "Type of Questions: SAQ\n"
        "Time: 2 hours 40 minutes\n"
        "Full marks: 70"
    )
    pages = [{"page_number": 1, "text": exam_text}]
    extracted_title = DocumentParser.extract_document_title("exam_paper.pdf", pages)
    assert "North East Nursing College" not in extracted_title
    assert "B-431 Midwifery" in extracted_title
    assert "Gynecological and Obstetrical Nursing" in extracted_title

def test_pipeline_title_and_metadata_routing():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "qdrant_db")
        bm25_path = os.path.join(tmpdir, "bm25.pkl")
        meta_path = os.path.join(tmpdir, "meta.pkl")

        pipeline = AdvancedRAGPipeline(
            storage_path=db_path,
            collection_name="test_chunks",
            bm25_path=bm25_path,
            metadata_store_path=meta_path
        )

        # Populate metadata & BM25 store manually for unit verification
        meta = DocumentMetadata(
            document_id="DOC_101",
            document_name="B413 Midwifery, P-II (Gynecological & Obstetrical Nursing).pdf",
            title="B413 Midwifery, P-II (Gynecological & Obstetrical Nursing)",
            total_pages=12,
            first_page_text="B413 Midwifery, P-II (Gynecological & Obstetrical Nursing)\nDepartment of Nursing.",
            summary="Midwifery and Gynecological nursing study notes."
        )
        pipeline.doc_metadata_store.add_document(meta)

        chunk1 = DocumentChunk(
            text="B413 Midwifery, P-II (Gynecological & Obstetrical Nursing). Course code B413.",
            metadata=ChunkMetadata(
                document_id="DOC_101",
                document_name="B413 Midwifery, P-II (Gynecological & Obstetrical Nursing).pdf",
                page_number=1,
                section="Title Section",
                chunk_id="DOC_101_001_01"
            )
        )
        pipeline.bm25_store.index_chunks([chunk1])

        # Test Title Query
        res_title = pipeline.query("What's the title of this PDF?")
        assert "B413 Midwifery" in res_title.answer
        assert res_title.query_intent == "metadata"

        # Test Page Count Query
        res_pages = pipeline.query("How many pages are in this document?")
        assert "12" in res_pages.answer
        assert res_pages.query_intent == "metadata"

        # Test Summary Query
        res_sum = pipeline.query("Summarize this document")
        assert len(res_sum.answer) > 10
        assert res_sum.query_intent in ["summary", "list"]
