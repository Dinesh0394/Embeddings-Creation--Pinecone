# vectorstore.py

from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from pinecone import Pinecone, ServerlessSpec

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def init_pinecone():
    # Extract cloud and region from PINECONE_ENV
    cloud, region = PINECONE_ENV.split("-", 1)

    # Create Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 outputs 384-dim vectors
            metric="cosine",
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )

    # Return the index client
    return pc.Index(PINECONE_INDEX_NAME)

def store_in_pinecone(chunks, embedder):
    index = init_pinecone()
    vectorstore = LangchainPinecone.from_documents(
        documents=chunks,
        embedding=embedder,
        index_name=PINECONE_INDEX_NAME
    )
    return vectorstore