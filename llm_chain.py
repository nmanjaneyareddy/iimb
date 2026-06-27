import os
import streamlit as st
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

def _get_secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None

def _make_llm():
    deepseek_key = _get_secret("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    openai_key = _get_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if deepseek_key:
        return ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2, max_tokens=512, api_key=deepseek_key, base_url=DEFAULT_BASE_URL)
    if openai_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=512, api_key=openai_key)
    raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in Streamlit secrets or environment variables")

class SimpleRetrievalQA:
    def __init__(self, retriever, llm, k=4):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Use the context to answer clearly and concisely. If the answer is not in the context, say you do not know.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs.get("input") or inputs.get("question") or inputs.get("query")
        if not question:
            return {"answer": ""}

        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(question)
        else:
            docs = self.retriever.get_relevant_documents(question)
        docs = docs[: self.k]

        context_parts: List[str] = [getattr(doc, "page_content", "") for doc in docs]
        context = "\n\n".join([part for part in context_parts if part])
        prompt_text = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_text)
        answer = getattr(response, "content", str(response))
        return {"answer": answer}

def setup_qa_chain(vectorstore, k=4):
    if vectorstore is None:
        raise ValueError("vectorstore cannot be None")
    llm = _make_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return SimpleRetrievalQA(retriever, llm, k=k)
