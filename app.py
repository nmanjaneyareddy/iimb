import streamlit as st
from vectorstore import load_or_create_vector_store
from llm_chain import setup_qa_chain

st.set_page_config(page_title="RAG QA App", layout="wide")
st.title("RAG Question Answering App")

try:
    vectorstore = load_or_create_vector_store()
    qa_chain = setup_qa_chain(vectorstore)
except Exception as exc:
    st.error("Startup failed: " + type(exc).__name__ + ": " + str(exc))
    st.stop()

question = st.text_input("Ask a question about your documents")

if question:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain.invoke({"input": question})
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            st.subheader("Answer")
            st.write(answer)
        except Exception as exc:
            st.error("Query failed: " + type(exc).__name__ + ": " + str(exc))
