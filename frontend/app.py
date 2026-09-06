import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="Local RAG System",
    page_icon="📚",
    layout="wide"
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

# --- Sidebar: Document Ingestion & Parameters ---
with st.sidebar:
    st.header("📄 Ingestion & Settings")

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt", "md"],
        help="Upload files to parse, chunk, embed, and index into Qdrant & BM25."
    )

    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Parsing, embedding, and indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = requests.post(f"{API_BASE_URL}/documents/upload", files=files)

                    if resp.status_code == 200:
                        data = resp.json()["details"]
                        st.success(f"Indexed: {data['filename']}")
                        st.caption(f"Pages: {data.get('pages_parsed', 1)} | Chunks: {data['chunks_indexed']}")
                        st.session_state.indexed_docs.append(data["filename"])
                    else:
                        st.error(f"Upload failed: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.divider()

    st.subheader("⚙️ Retrieval Parameters")
    retrieval_candidates = st.slider("Hybrid Retrieval Candidates", min_value=3, max_value=20, value=8)
    top_n_rerank = st.slider("Cross-Encoder Top-N", min_value=1, max_value=5, value=2)

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# Main Workspace (Completely outside the sidebar block)
# ==============================================================================
st.title("📚 Fully Local RAG System")
st.caption("Hybrid RRF (BGE-M3 + BM25) ➔ Neural Reranker (BGE) ➔ Qwen2.5-0.5B-Instruct")

# Render message history in main screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("🔍 Citations & Sources"):
                for idx, c in enumerate(msg["citations"], 1):
                    st.markdown(
                        f"**[{idx}] {c['document_name']}** (Page: {c['page_number']}, Section: `{c['section']}`)\n"
                        f"*Chunk ID: `{c['chunk_id']}`*"
                    )
        if "latency" in msg and msg["latency"]:
            lat = msg["latency"]
            st.caption(
                f"⏱️ **Retrieval:** {lat['retrieval']} ms | "
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
                payload = {
                    "query": prompt,
                    "retrieval_candidates": retrieval_candidates,
                    "top_n_rerank": top_n_rerank
                }
                resp = requests.post(f"{API_BASE_URL}/rag/query", json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])
                    latency = {
                        "retrieval": data.get("retrieval_latency_ms", 0.0),
                        "rerank": data.get("rerank_latency_ms", 0.0),
                        "gen": data.get("generation_latency_ms", 0.0),
                        "total": data.get("total_latency_ms", 0.0)
                    }

                    st.markdown(answer)

                    if citations:
                        with st.expander("🔍 Citations & Sources"):
                            for idx, c in enumerate(citations, 1):
                                st.markdown(
                                    f"**[{idx}] {c['document_name']}** (Page: {c['page_number']}, Section: `{c['section']}`)\n"
                                    f"*Chunk ID: `{c['chunk_id']}`*"
                                )

                    st.caption(
                        f"⏱️ **Retrieval:** {latency['retrieval']} ms | "
                        f"**Rerank:** {latency['rerank']} ms | "
                        f"**Gen:** {latency['gen']} ms | "
                        f"**Total:** {latency['total']} ms"
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "latency": latency
                    })
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not reach the FastAPI server. Make sure Uvicorn is active on http://127.0.0.1:8000.")