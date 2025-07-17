# config.py

import os

# Load API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Example: "gcp-starter"
PINECONE_INDEX_NAME = "cricket-rules-index"

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
