# # vectorstore.py

# import pinecone
# from langchain_community.vectorstores import Pinecone
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, CHUNK_SIZE, CHUNK_OVERLAP

# def chunk_documents(documents):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     return splitter.split_documents(documents)

# def init_pinecone():
#     pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
#     if PINECONE_INDEX_NAME not in pinecone.list_indexes():
#         pinecone.create_index(PINECONE_INDEX_NAME, dimension=384, metric="cosine")
#     index = pinecone.Index(PINECONE_INDEX_NAME)
#     return index

# def store_in_pinecone(chunks, embedder):
#     index = init_pinecone()
#     vectorstore = Pinecone.from_documents(chunks, embedder, index_name=PINECONE_INDEX_NAME)
#     return vectorstore



# vectorstore.py

import Pinecone
from langchain_community.vectorstores import Pinecone # Keep this import for now, as Pinecone.from_documents uses it
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, CHUNK_SIZE, CHUNK_OVERLAP
import os # It's good practice to import os if you are getting env vars from it, though config handles it.

def chunk_documents(documents):
    splitter = RecursiveCharacterTextTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def init_pinecone():
    # Initialize the Pinecone client instance
    # Pass environment to the Pinecone constructor for pod-based indexes
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    # Check if the index exists using the new client instance
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        # Create the index using the new client instance
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # As per your original code
            metric="cosine" # As per your original code
            # If you were using a serverless index, you'd add:
            # spec=pinecone.ServerlessSpec(cloud='aws', region='us-west-2')
            # but based on your PINECONE_ENV, it seems like a pod-based setup.
        )
    
    # Get the index object from the new client instance
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

def store_in_pinecone(chunks, embedder):
    index = init_pinecone()
    # This line uses Pinecone from langchain_community, which expects a Pinecone Index object
    vectorstore = Pinecone.from_documents(chunks, embedder, index_name=PINECONE_INDEX_NAME)
    return vectorstore