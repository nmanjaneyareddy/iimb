import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}

def _load_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()

    if ext in {".html", ".htm"}:
        return BSHTMLLoader(str(path), bs_kwargs={"features": "html.parser"}).load()

    if ext in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()

    return []

def load_documents(data_dir=DATA_DIR):
    docs = []

    data_path = Path(data_dir)

    if not data_path.is_absolute():
        data_path = BASE_DIR / data_path

    if not data_path.is_dir():
        raise FileNotFoundError("Data directory not found: " + str(data_path))

    for root, _, files in os.walk(data_path):
        for file_name in sorted(files):
            path = Path(root) / file_name

            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.extend(_load_file(path))

    if not docs:
        raise ValueError("No supported documents found in " + str(data_path) + ".")

    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
