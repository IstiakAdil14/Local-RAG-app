# 📚 Fully Local & Offline RAG System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://local-rag-app-fpnbg4fqmarllxjiu2qm3s.streamlit.app/)

**🌐 Deployed Live App**: [https://local-rag-app-fpnbg4fqmarllxjiu2qm3s.streamlit.app/](https://local-rag-app-fpnbg4fqmarllxjiu2qm3s.streamlit.app/)

An end-to-end, privacy-first, fully local Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Qdrant**, **Streamlit**, and Hugging Face transformer models. Running completely offline with 0% external API dependencies, this system ingests multi-format documents, executes hybrid sparse-dense retrieval, reranks results with a neural cross-encoder, and generates grounded, cited answers using local LLMs.

---

## 🌟 Key Features

* **🔒 100% Local & Privacy-Focused**: Runs completely offline without external APIs or cloud dependencies (no OpenAI/Anthropic/Google keys required).
* **📄 Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.txt`, and `.md` files with OCR fallback capabilities (using PyMuPDF, Docling, python-docx, and PyTesseract).
* **🔀 Hybrid Sparse-Dense Retrieval**: Combines dense semantic vectors (**BAAI/bge-m3** via Qdrant) and sparse keyword matching (**BM25**) using **Reciprocal Rank Fusion (RRF)**.
* **🎯 Neural Cross-Encoder Reranking**: Re-evaluates top-K candidates using **BAAI/bge-reranker-base** to improve contextual precision.
* **🧠 Grounded Generation & Strict Anti-Hallucination**: Employs **Qwen/Qwen2.5-0.5B-Instruct** (or 1.5B) to generate strict, fact-grounded answers based solely on retrieved context.
* **📌 Transparent Citations**: Tracks and displays exact citations (document name, page number, section heading, and chunk ID).
* **⏱️ Granular Latency Profiling**: Real-time breakdown of retrieval, reranking, generation, and total latency.
* **⚡ FastAPI Backend**: High-performance RESTful API endpoints for ingestion, health checks, and querying.
* **🎨 Interactive Streamlit UI**: User-friendly chat interface with file upload, parametric sliders for candidate tuning, chat memory management, and expandable source details.

---

## 🏗️ System Architecture

```
                                    +-----------------------+
                                    |  Uploaded Document    |
                                    | (.pdf, .docx, .txt)   |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | Parsing & Chunking    |
                                    | (PyMuPDF/Docling)     |
                                    +-----------+-----------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
            +-----------------------+                       +-----------------------+
            | Dense Embedding       |                       | Sparse Indexing       |
            | (BAAI/bge-m3)         |                       | (BM25 Engine)         |
            +-----------+-----------+                       +-----------+-----------+
                        |                                               |
                        v                                               v
            +-----------------------+                       +-----------------------+
            | Local Qdrant Vector   |                       | BM25 Pickle Storage   |
            | Database              |                       |                       |
            +-----------+-----------+                       +-----------+-----------+
                        |                                               |
                        +-----------------------+-----------------------+
                                                |
                                                v
                                    +-----------------------+
                                    | Hybrid Search (RRF)   |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | Neural Reranker       |
                                    | (bge-reranker-base)   |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | Local SLM Generator   |
                                    | (Qwen2.5-0.5B)        |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | Answer & Citations    |
                                    +-----------------------+
```

---

## 📁 Repository Structure

```
local-rag-system/
├── backend/
│   ├── app/
│   │   ├── api/             # API routes and controllers
│   │   ├── core/            # Core configuration & settings
│   │   ├── ingestion/       # Document parsing, chunking & service
│   │   │   ├── chunker.py   # Hierarchical & semantic chunker
│   │   │   ├── parser.py    # Document text parser (.pdf, .docx, etc.)
│   │   │   └── service.py   # Ingestion orchestration
│   │   ├── rag/             # Core RAG components
│   │   │   ├── bm25_search.py   # BM25 sparse search engine
│   │   │   ├── embeddings.py    # BGE-M3 local embedding engine
│   │   │   ├── generator.py     # Local Qwen LLM generator
│   │   │   ├── hybrid_search.py # RRF sparse-dense fusion
│   │   │   ├── pipeline.py      # Baseline & Advanced RAG pipelines
│   │   │   ├── reranker.py      # Cross-Encoder reranker
│   │   │   └── vector_search.py # Qdrant local vector store integration
│   │   ├── schemas/         # Pydantic models & request/response schemas
│   │   └── main.py          # FastAPI application entry point
│   ├── requirements.txt     # Backend Python dependencies
│   └── tests/               # Unit and integration tests
├── data/                    # Persistent storage for Qdrant DB & BM25 index
├── frontend/
│   └── app.py               # Streamlit web application
├── models/                  # Optional local model cache directory
├── notebooks/               # RAG experimentation and exploratory notebooks
├── scripts/                 # Verification and setup scripts
│   ├── download_models.py   # Pre-fetch & cache Hugging Face models
│   ├── verify_pipeline.py   # End-to-end RAG pipeline test
│   └── verify_phase*.py     # Component verification scripts
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

* **Python**: 3.10 or higher
* **RAM**: 8 GB minimum (16 GB recommended for GPU/CPU inference)
* **GPU** *(Optional)*: NVIDIA GPU with CUDA support for accelerated inference.

---

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/IstiakAdil14/Local-RAG-app.git
   cd Local-RAG-app
   ```

2. **Set up a Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   pip install streamlit
   ```

4. **Pre-download Local Models** *(Recommended)*:
   Cache the LLM and embedding models locally before running the system:
   ```bash
   python scripts/download_models.py
   ```

---

## 🏃 Running the Application

### 1. Launch the FastAPI Backend
Start the backend server on `http://127.0.0.1:8000`:
```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
* Interactive API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`

### 2. Launch the Streamlit Frontend
In a separate terminal (with the virtual environment activated), launch the UI:
```bash
streamlit run frontend/app.py
```
* Access the web UI in your browser at `http://localhost:8501`.

---

## 📡 API Reference

### Health Check
`GET /api/v1/health`
* **Response**:
  ```json
  {
    "status": "online",
    "mode": "local_offline"
  }
  ```

### Document Upload
`POST /api/v1/documents/upload`
* **Form Data**: `file` (Multipart file: `.pdf`, `.docx`, `.txt`, `.md`)
* **Response**:
  ```json
  {
    "message": "Document indexed successfully",
    "details": {
      "filename": "sample.pdf",
      "pages_parsed": 5,
      "chunks_indexed": 24
    }
  }
  ```

### Query RAG Pipeline
`POST /api/v1/rag/query`
* **Request Body**:
  ```json
  {
    "query": "What are the main findings in the document?",
    "retrieval_candidates": 10,
    "top_n_rerank": 3
  }
  ```
* **Response**:
  ```json
  {
    "query": "What are the main findings in the document?",
    "answer": "The main findings state that...",
    "citations": [
      {
        "document_name": "sample.pdf",
        "page_number": 2,
        "section": "Results",
        "chunk_id": "chunk_004"
      }
    ],
    "retrieval_latency_ms": 45.2,
    "rerank_latency_ms": 18.7,
    "generation_latency_ms": 320.5,
    "total_latency_ms": 384.4
  }
  ```

---

## 🧪 Verification & Testing

Verify that all components (embeddings, vector search, BM25, hybrid fusion, reranker, and generator) are working correctly:

```bash
# Run complete RAG pipeline verification
python scripts/verify_pipeline.py

# Run component-specific phase verifications
python scripts/verify_phase1.py  # Ingestion & Parsing
python scripts/verify_phase2.py  # Dense Embeddings & Vector Store
python scripts/verify_phase3.py  # Sparse BM25 Search
python scripts/verify_phase4.py  # Baseline RAG Pipeline
python scripts/verify_phase5.py  # Hybrid RRF Search
python scripts/verify_phase6.py  # Cross-Encoder Reranker
python scripts/verify_phase7.py  # Advanced RAG Pipeline
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
