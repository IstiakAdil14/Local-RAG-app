import os
import pytest
import tempfile
from app.rag.pipeline import AdvancedRAGPipeline
from app.schemas.document import DocumentMetadata, DocumentChunk, ChunkMetadata
from app.ingestion.parser import DocumentParser

# GLOBAL RAG EVALUATION BENCHMARK TEST MATRIX
BENCHMARK_MATRIX = [
    # 1. METADATA QUERIES
    {"query": "What is the title of this PDF?", "category": "Metadata", "expected_keyword": "B-431 Midwifery", "should_abstain": False},
    {"query": "How many pages are in this document?", "category": "Metadata", "expected_keyword": "12", "should_abstain": False},
    {"query": "What is the filename of the document?", "category": "Metadata", "expected_keyword": "B431_Midwifery_Exam.pdf", "should_abstain": False},

    # 2. FACT RETRIEVAL QUERIES
    {"query": "TELL ME ABOUT HER NURSING EXPERIENCES", "category": "Fact", "expected_keyword": "Emergency Ward", "should_abstain": False},
    {"query": "What is the subject code for midwifery?", "category": "Fact", "expected_keyword": "B-431", "should_abstain": False},
    {"query": "What paper is Gynecological and Obstetrical Nursing?", "category": "Fact", "expected_keyword": "Paper II", "should_abstain": False},

    # 3. LIST QUERIES
    {"query": "List the wards in her experience", "category": "List", "expected_keyword": "NICU", "should_abstain": False},
    {"query": "What are all the sections in the document?", "category": "List", "expected_keyword": "Nursing", "should_abstain": False},

    # 4. SUMMARY QUERIES
    {"query": "Give me an overview of the document", "category": "Summary", "expected_keyword": "nursing", "should_abstain": False},
    {"query": "What is this file about?", "category": "Summary", "expected_keyword": "Midwifery", "should_abstain": False},

    # 5. NUMERICAL QUERIES
    {"query": "What are the full marks for the exam?", "category": "Numerical", "expected_keyword": "70", "should_abstain": False},
    {"query": "What is the time allowed for the exam?", "category": "Numerical", "expected_keyword": "2 hours 40 minutes", "should_abstain": False},

    # 6. UNANSWERABLE / ABSTENTION QUERIES (FALSE-ABSTENTION & HALLUCINATION TEST)
    {"query": "What is the author's blood group?", "category": "Unanswerable", "expected_keyword": "does not specify", "should_abstain": True},
    {"query": "What is her passport number?", "category": "Unanswerable", "expected_keyword": "does not specify", "should_abstain": True},
    {"query": "What is the candidate's favorite color?", "category": "Unanswerable", "expected_keyword": "does not specify", "should_abstain": True},
]

def setup_benchmark_pipeline(tmpdir):
    db_path = os.path.join(tmpdir, "qdrant_db")
    bm25_path = os.path.join(tmpdir, "bm25.pkl")
    meta_path = os.path.join(tmpdir, "meta.pkl")

    pipeline = AdvancedRAGPipeline(
        storage_path=db_path,
        collection_name="benchmark_chunks",
        bm25_path=bm25_path,
        metadata_store_path=meta_path
    )

    # Academic Exam Document
    exam_meta = DocumentMetadata(
        document_id="DOC_EXAM_001",
        document_name="B431_Midwifery_Exam.pdf",
        title="B-431 Midwifery & Obstetrical Nursing - Paper II: Gynecological and Obstetrical Nursing",
        total_pages=12,
        first_page_text=(
            "North East Nursing College, Sylhet\n"
            "4th Year B.Sc. in Nursing Midterm Examination, July-2025\n"
            "Subject: B-431 Midwifery & Obstetrical Nursing\n"
            "Paper II: Gynecological and Obstetrical Nursing\n"
            "Type of Questions: SAQ\n"
            "Time: 2 hours 40 minutes\n"
            "Full marks: 70\n"
            "[Instruction: Use separate answer scripts for each group]"
        ),
        summary="Midterm examination question paper for 4th Year B.Sc. in Nursing on Midwifery & Obstetrical Nursing."
    )
    pipeline.doc_metadata_store.add_document(exam_meta)

    # Indexed Chunks
    chunk1 = DocumentChunk(
        text=(
            "North East Nursing College, Sylhet\n"
            "4th Year B.Sc. in Nursing Midterm Examination, July-2025\n"
            "Subject: B-431 Midwifery & Obstetrical Nursing\n"
            "Paper II: Gynecological and Obstetrical Nursing\n"
            "Type of Questions: SAQ\n"
            "Time: 2 hours 40 minutes\n"
            "Full marks: 70"
        ),
        metadata=ChunkMetadata(
            document_id="DOC_EXAM_001",
            document_name="B431_Midwifery_Exam.pdf",
            page_number=1,
            section="Header Section",
            chunk_id="DOC_EXAM_001_001"
        )
    )

    chunk2 = DocumentChunk(
        text=(
            "CAREER OBJECTIVE: Confident nurse.\n"
            "EXPERIENCE:\n"
            "* Emergency Ward * Surgery Ward * Medicine Ward\n"
            "* Post-Operative Ward\n"
            "* NICU, ICU"
        ),
        metadata=ChunkMetadata(
            document_id="DOC_EXAM_001",
            document_name="B431_Midwifery_Exam.pdf",
            page_number=2,
            section="Experience Section",
            chunk_id="DOC_EXAM_001_002"
        )
    )

    pipeline.bm25_store.index_chunks([chunk1, chunk2])
    return pipeline

def test_global_rag_benchmark_suite():
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = setup_benchmark_pipeline(tmpdir)
        try:
            passed_count = 0
            total_count = len(BENCHMARK_MATRIX)

            print("\n===== RUNNING GLOBAL RAG BENCHMARK EVALUATION =====")
            for item in BENCHMARK_MATRIX:
                query = item["query"]
                cat = item["category"]
                expected = item["expected_keyword"]
                should_abstain = item["should_abstain"]

                res = pipeline.query(query)
                ans = res.answer

                if should_abstain:
                    assert "does not specify" in ans.lower() or "not specify" in ans.lower(), f"Failed abstention check for query: {query}. Answer was: {ans}"
                else:
                    assert expected.lower() in ans.lower(), f"Failed accuracy check for query: [{query}] ({cat}). Expected keyword '{expected}' in answer: '{ans}'"

                passed_count += 1

            accuracy_pct = (passed_count / float(total_count)) * 100.0
            print(f"\n✅ BENCHMARK ACCURACY: {accuracy_pct:.1f}% ({passed_count}/{total_count} Passed)")
            assert accuracy_pct == 100.0
        finally:
            try:
                pipeline.vector_store.client.close()
            except Exception:
                pass
