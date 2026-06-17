FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST before other requirements
# This prevents pip from pulling the full CUDA version (~4GB)
# CPU-only torch is ~500MB — sufficient for sentence-transformers inference
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model during build
RUN python -m spacy download en_core_web_lg

# Pre-download HuggingFace embedding model during build
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p chroma_db data/sample_docs

# Use Railway's dynamic PORT variable
# Falls back to 8000 for local docker-compose testing
EXPOSE ${PORT:-8000}

# Ensure pydantic-settings reads from os.environ, not just .env file
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]