import os
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}

def _load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext in {".html", ".htm"}:
        return BSHTMLLoader(path, bs_kwargs={"features": "html.parser"}).load()
    if ext in {".txt", ".md"}:
        return TextLoader(path, encoding="utf-8").load()
    return []

def load_documents(data_dir=DATA_DIR):
    docs = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError("Data directory not found: " + data_dir)

    for root, _, files in os.walk(data_dir):
        for file_name in sorted(files):
            path = os.path.join(root, file_name)
            if os.path.splitext(file_name)[1].lower() in SUPPORTED_EXTENSIONS:
                docs.extend(_load_file(path))

    if not docs:
        raise ValueError("No supported documents found in " + data_dir + ".")
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
