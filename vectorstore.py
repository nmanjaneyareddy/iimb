# vectorstore.py
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "iimb_library_details.pdf"

@st.cache_resource(show_spinner="Loading PDF and building vector index...")
def load_or_create_vector_store():
    """
    Loads the IGIDR library PDF, splits it into chunks,
    creates embeddings, and builds an in-memory FAISS index.
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError("PDF knowledge base not found: " + str(PDF_PATH))

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    if not documents:
        raise ValueError("No text could be loaded from PDF: " + str(PDF_PATH))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created from PDF: " + str(PDF_PATH))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
