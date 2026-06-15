FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker layer caching means this layer
# only rebuilds when requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model during build — not at runtime
# Avoids 400MB download on container startup
RUN python -m spacy download en_core_web_lg

# Pre-download HuggingFace embedding model during build
# This bakes the model into the image (~90MB)
# Avoids download on first request which would timeout health checks
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
# Done after dependencies to maximise layer cache hits
COPY . .

# Create directories that must exist at runtime
RUN mkdir -p chroma_db data/sample_docs

# Expose port
EXPOSE 8000

# Run the application
# Use 0.0.0.0 to accept connections from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]