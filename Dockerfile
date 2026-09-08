# Use lightweight Python 3.11 base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models/huggingface \
    TRANSFORMERS_CACHE=/app/models/huggingface

# Install system dependencies (including Tesseract OCR & Poppler for PDF parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/models/huggingface /app/data/qdrant_db

# Copy backend requirements first to leverage Docker cache
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir streamlit

# Copy application files
COPY backend /app/backend
COPY frontend /app/frontend
COPY scripts /app/scripts
COPY .env.example /app/.env.example
COPY start.sh /app/start.sh

# Grant execution permission to startup script
RUN chmod +x /app/start.sh

# Pre-download and cache local models into the image so HF Space boots instantly
RUN python /app/scripts/download_models.py

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run entrypoint script
CMD ["/bin/bash", "/app/start.sh"]
