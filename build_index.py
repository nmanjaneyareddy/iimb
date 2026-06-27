from loaders import load_documents, split_documents
from vectorstore import create_vector_store

docs = load_documents()
chunks = split_documents(docs)
create_vector_store(chunks)
print("FAISS index created successfully")
