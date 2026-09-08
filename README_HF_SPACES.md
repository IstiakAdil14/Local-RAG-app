# 🚀 Deploying Local RAG System to Hugging Face Spaces (100% Free Cloud)

This guide walks you through deploying your **Local RAG System** to **Hugging Face Spaces** for free using the provided `Dockerfile`.

---

## 🌟 Why Hugging Face Spaces?
- **100% Free**: Includes **16 GB RAM**, 2 vCPUs, and 50 GB persistent storage on the free tier.
- **Pre-cached Models**: The `Dockerfile` pre-downloads `Qwen/Qwen2.5-0.5B-Instruct`, `BAAI/bge-m3`, and `BAAI/bge-reranker-base` so your Space boots quickly.
- **Zero Cost Vector DB**: Connects directly to **Qdrant Cloud Free Tier**.

---

## 📋 Step-by-Step Deployment Guide

### Step 1: Create a Free Hugging Face Space
1. Log into your [Hugging Face Account](https://huggingface.co/) (create one for free if needed).
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Set your Space configuration:
   - **Space Name**: `local-rag-app` (or any name you prefer)
   - **License**: `mit`
   - **Select the Space SDK**: Choose **Docker**
   - **Choose a Docker Template**: Choose **Blank**
   - **Space Hardware**: Select **CPU Basic • 2 vCPU • 16 GB RAM • Free**
   - **Visibility**: Public or Private
4. Click **Create Space**.

---

### Step 2: Add Secret Environment Variables (Optional but Recommended)
If you want your Space to connect to your Qdrant Cloud Cluster:
1. In your new Hugging Face Space, click on the **Settings** tab.
2. Scroll down to **Variables and secrets**.
3. Add New Secrets:
   - `QDRANT_URL`: `https://a4cdb6c0-98b5-4ce9-98ce-45300522a7cd.sa-east-1-0.aws.cloud.qdrant.io:6333`
   - `QDRANT_API_KEY`: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YjE2YzFhZjItNGZhZS00YzQ1LWE3NzgtZWU0ODRkNmE5NDBiIn0.MA_pDHqpN4X9VLI6pAk1a31JtYkxs7rEQpocC3LUpz8`
   - `QDRANT_COLLECTION_NAME`: `production_coll`

---

### Step 3: Push Your Project Code to Hugging Face
You can push your repository to Hugging Face using standard Git commands.

#### Using Git:
```bash
# Initialize git if needed
git init

# Add Hugging Face Space remote (replace username/space-name with yours)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/local-rag-app

# Commit all deployment files
git add .
git commit -m "Deploy Local RAG system to HF Spaces"

# Push code to Hugging Face
git push hf main --force
```

---

## 🔍 Verification & Access
1. Once pushed, Hugging Face will automatically trigger `docker build`.
2. View the build logs under the **App** or **Logs** tab.
3. Once building completes, your web UI will be live at:
   `https://huggingface.co/spaces/YOUR_USERNAME/local-rag-app`
4. Test by uploading a `.pdf`, `.docx`, or `.txt` file and asking questions in the chat interface!
