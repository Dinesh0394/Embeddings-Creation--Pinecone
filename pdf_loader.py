# pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages
