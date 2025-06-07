# 📚 PDF Q&A — Intelligent Document Query App

This lightweight app offers a simple implementation of an agentic Retrieval-Augmented Generation (RAG) system. It allows you to upload a document and ask questions, routing your query to either a summarization or retrieval engine based on the intent. Built to demonstrate the basics of multi-tool LLM systems using LlamaIndex and Streamlit.



---

## 🔮 Features

- 🧾 Upload any PDF and ask **natural language** questions
- 🧠 Dual-indexing with:
  - **Summarization Index** for overviews
  - **Vector Index** for semantic fact retrieval
- 🧭 Uses **RouterQueryEngine** to automatically pick the best tool
- 📦 Fully modular — plug in **any LLM** or **embedding model** you love
- 🎈 Built with **Streamlit**, **LlamaIndex**, and **HuggingFace Embeddings**

---

## 🧠 How It Works

1. Upload a PDF through the frontend.
2. The file is parsed and chunked using `SentenceSplitter`.
3. Two indexes are created:
   - `SummaryIndex`: captures document-wide context for summarization
   - `VectorStoreIndex`: supports retrieval and semantic search
4. A `RouterQueryEngine` decides whether the query is best handled via:
   - 📄 Summary Tool → "Give me a TL;DR"
   - 🔎 Vector Tool → "What’s the author’s stance on X?"
5. Boom. Instant insights.

---

## 🚀 Getting Started

### 📦 Installation

First, make sure you have Python 3.10+.

```bash
pip install -r requirements.txt

