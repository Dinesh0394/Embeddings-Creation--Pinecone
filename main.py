# main.py

from pdf_loader import load_pdf
from embedder import load_embedder
from vectorstore import chunk_documents, store_in_pinecone

PDF_PATH = "cricket_rules.pdf"

def main():
    print("[✓] Loading PDF...")
    pages = load_pdf(PDF_PATH)

    print("[✓] Chunking text...")
    chunks = chunk_documents(pages)

    print("[✓] Loading embedding model...")
    embedder = load_embedder()

    print("[✓] Storing embeddings in Pinecone...")
    store_in_pinecone(chunks, embedder)

    print("[🚀] Success! Embeddings stored in Pinecone.")

if __name__ == "__main__":
    main()
