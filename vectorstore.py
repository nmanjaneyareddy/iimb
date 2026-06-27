# vectorstore.py
from pathlib import Path

import streamlit as st
import fitz

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "igidr_library_details.pdf"

def load_pdf_with_links(pdf_path):
    documents = []

    pdf = fitz.open(str(pdf_path))

    for page_index, page in enumerate(pdf):
        text = page.get_text("text")

        links = []
        for link in page.get_links():
            uri = link.get("uri")
            if uri:
                links.append(uri)

        unique_links = []
        for link in links:
            if link not in unique_links:
                unique_links.append(link)

        if unique_links:
            text += "\n\nLinks found on this page:\n"
            for link in unique_links:
                text += "- " + link + "\n"

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_index + 1,
                    "links": unique_links,
                },
            )
        )

    pdf.close()
    return documents

@st.cache_resource(show_spinner="Loading PDF links and building vector index...")
def load_or_create_vector_store():
    """
    Loads the IGIDR library PDF, extracts text plus embedded hyperlinks,
    splits it into chunks, creates embeddings, and builds an in-memory FAISS index.
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError("PDF knowledge base not found: " + str(PDF_PATH))

    documents = load_pdf_with_links(PDF_PATH)

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
