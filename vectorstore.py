import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loaders import load_documents, split_documents

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner="Loading documents and building vector index...")
def load_or_create_vector_store():
    docs = load_documents()
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("No document chunks were created. Check the files in the data directory.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def create_vector_store(docs, index_path="faiss_index"):
    if not docs:
        raise ValueError("No documents supplied to create_vector_store.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    return vectorstore
