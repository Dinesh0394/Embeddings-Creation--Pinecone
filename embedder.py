# embedder.py

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

def load_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
