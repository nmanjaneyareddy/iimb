# RAG Chatbot using Streamlit, FAISS, PDF, and HTML

A lightweight Retrieval-Augmented Generation app that answers questions from documents in the `data/` directory.

## Supported files

- PDF
- HTML / HTM
- TXT
- Markdown

## Run locally

```bash
git clone https://github.com/nmanjaneyareddy/librag.git
cd librag
pip install -r requirements.txt
streamlit run app.py
```

## API keys

Set one of these in Streamlit secrets or environment variables:

```toml
DEEPSEEK_API_KEY = "your-key"
OPENAI_API_KEY = "your-key"
```

If `DEEPSEEK_API_KEY` is present, the app uses `deepseek-chat`. Otherwise it falls back to OpenAI `gpt-4o-mini`.

## Build a local FAISS index manually

```bash
python build_index.py
```
