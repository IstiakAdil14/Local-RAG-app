import streamlit as st
import requests
import asyncio
import io
import os
import sys
from pathlib import Path

# Fix import path for Streamlit Cloud & local execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
backend_path = project_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="Local RAG System",
    page_icon=":material/auto_stories:",
    layout="wide"
)

# --- Direct Python Pipeline Initialization (Fallback for Cloud / Standalone) ---
@st.cache_resource
def get_direct_services():
    try:
        from app.rag.pipeline import AdvancedRAGPipeline
        from app.ingestion.service import DocumentIngestionService
        from app.core.config import settings

        rag_pipeline = AdvancedRAGPipeline(
            storage_path=settings.QDRANT_STORAGE_PATH,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            bm25_path=settings.BM25_INDEX_PATH,
            qdrant_url=settings.QDRANT_URL,
            qdrant_api_key=settings.QDRANT_API_KEY,
            generator_model=settings.GENERATOR_MODEL_ID,
            reranker_model=settings.RERANKER_MODEL_ID
        )
        ingestion_service = DocumentIngestionService(
            embedder=rag_pipeline.embedder,
            vector_store=rag_pipeline.vector_store,
            bm25_store=rag_pipeline.bm25_store
        )
        return rag_pipeline, ingestion_service
    except Exception as e:
        st.error(f"Error initializing direct RAG services: {e}")
        return None, None

class UploadFileWrapper:
    def __init__(self, filename, bytes_io):
        self.filename = filename
        self.file = bytes_io

# Helper to ingest file (tries HTTP API first, falls back to direct Python)
def perform_ingest(uploaded_file):
    # Try HTTP API
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        resp = requests.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=5)
        if resp.status_code == 200:
            return resp.json()["details"]
    except Exception:
        pass

    # Direct Python fallback
    rag_pipeline, ingestion_service = get_direct_services()
    if ingestion_service is None:
        raise Exception("Direct ingestion service unavailable.")
    
    bytes_io = io.BytesIO(uploaded_file.getvalue())
    wrapper = UploadFileWrapper(uploaded_file.name, bytes_io)
    
    # Run async ingest safely
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(ingestion_service.ingest_file(wrapper))
    loop.close()
    return result

# Helper to query RAG (tries HTTP API first, falls back to direct Python)
def perform_query(prompt, retrieval_candidates, top_n_rerank):
    try:
        payload = {
            "query": prompt,
            "retrieval_candidates": retrieval_candidates,
            "top_n_rerank": top_n_rerank
        }
        resp = requests.post(f"{API_BASE_URL}/rag/query", json=payload, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "answer": data["answer"],
                "citations": data.get("citations", []),
                "latency": {
                    "retrieval": data.get("retrieval_latency_ms", 0.0),
                    "rerank": data.get("rerank_latency_ms", 0.0),
                    "gen": data.get("generation_latency_ms", 0.0),
                    "total": data.get("total_latency_ms", 0.0)
                }
            }
    except Exception:
        pass

    # Direct Python fallback
    rag_pipeline, _ = get_direct_services()
    if rag_pipeline is None:
        raise Exception("Direct RAG pipeline service unavailable.")

    resp_obj = rag_pipeline.query(
        user_query=prompt,
        retrieval_candidates=retrieval_candidates,
        top_n_rerank=top_n_rerank
    )

    citations = [
        {
            "document_name": c.document_name,
            "page_number": c.page_number,
            "section": c.section,
            "chunk_id": c.chunk_id
        }
        for c in resp_obj.citations
    ]

    return {
        "answer": resp_obj.answer,
        "citations": citations,
        "latency": {
            "retrieval": resp_obj.retrieval_latency_ms,
            "rerank": resp_obj.rerank_latency_ms,
            "gen": resp_obj.generation_latency_ms,
            "total": resp_obj.total_latency_ms
        }
    }

# Helper to reset database
def perform_reset():
    try:
        resp = requests.post(f"{API_BASE_URL}/documents/reset", timeout=5)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    _, ingestion_service = get_direct_services()
    if ingestion_service:
        ingestion_service.clear_database()
        return True
    return False

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

# --- Sidebar: Document Ingestion & Parameters ---
with st.sidebar:
    st.header("Ingestion & Settings", icon=":material/settings:")

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt", "md"],
        help="Upload files to parse, chunk, embed, and index into Qdrant & BM25."
    )

    if uploaded_file is not None:
        if st.button("Index Document", icon=":material/upload_file:", use_container_width=True):
            with st.spinner("Parsing, embedding, and indexing..."):
                try:
                    data = perform_ingest(uploaded_file)
                    st.success(f"Indexed: {data['filename']}")
                    st.caption(f"Pages: {data.get('pages_parsed', 1)} | Chunks: {data['chunks_indexed']}")
                    st.session_state.indexed_docs.append(data["filename"])
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    st.subheader("Retrieval Parameters", icon=":material/tune:")
    retrieval_candidates = st.slider("Hybrid Retrieval Candidates", min_value=3, max_value=20, value=5)
    top_n_rerank = st.slider("Cross-Encoder Top-N", min_value=1, max_value=5, value=2)

    st.divider()
    if st.button("Clear Chat History", icon=":material/cleaning_services:", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Knowledge Base", icon=":material/delete_forever:", use_container_width=True, help="Clear all indexed documents from vector database and BM25 index"):
        try:
            if perform_reset():
                st.session_state.indexed_docs = []
                st.session_state.messages = []
                st.success("Knowledge Base reset successfully!")
                st.rerun()
            else:
                st.error("Reset failed.")
        except Exception as e:
            st.error(f"Reset error: {e}")

# ==============================================================================
# Main Workspace
# ==============================================================================
st.title("Fully Local RAG System")
st.caption("Hybrid RRF (BGE-M3 + BM25) ➔ Neural Reranker (BGE) ➔ Qwen2.5-0.5B-Instruct")

# Render message history in main screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("Citations & Sources", icon=":material/source:"):
                for idx, c in enumerate(msg["citations"], 1):
                    st.markdown(
                        f"**[{idx}] {c['document_name']}** (Page: {c['page_number']}, Section: `{c['section']}`)\n"
                        f"*Chunk ID: `{c['chunk_id']}`*"
                    )
        if "latency" in msg and msg["latency"]:
            lat = msg["latency"]
            st.caption(
                f"**Retrieval:** {lat['retrieval']} ms | "
                f"**Rerank:** {lat['rerank']} ms | "
                f"**Gen:** {lat['gen']} ms | "
                f"**Total:** {lat['total']} ms"
            )

# Chat Input at the bottom of the main viewport
if prompt := st.chat_input("Ask a question about your indexed documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, and generating..."):
            try:
                result = perform_query(prompt, retrieval_candidates, top_n_rerank)
                answer = result["answer"]
                citations = result.get("citations", [])
                latency = result.get("latency", {})

                st.markdown(answer)

                if citations:
                    with st.expander("Citations & Sources", icon=":material/source:"):
                        for idx, c in enumerate(citations, 1):
                            st.markdown(
                                f"**[{idx}] {c['document_name']}** (Page: {c['page_number']}, Section: `{c['section']}`)\n"
                                f"*Chunk ID: `{c['chunk_id']}`*"
                            )

                st.caption(
                    f"**Retrieval:** {latency.get('retrieval', 0.0)} ms | "
                    f"**Rerank:** {latency.get('rerank', 0.0)} ms | "
                    f"**Gen:** {latency.get('gen', 0.0)} ms | "
                    f"**Total:** {latency.get('total', 0.0)} ms"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "latency": latency
                })

            except Exception as e:
                st.error(f"Query processing error: {e}")