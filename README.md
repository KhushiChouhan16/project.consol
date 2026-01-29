# 📄 Intelligent PDF Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDFs and ask questions strictly based on the document content.

## 🚀 Features
- Upload PDF files
- Ask questions from the PDF
- Context-aware answers using RAG
- Chat history per PDF
- Local LLM using Ollama (Gemma / LLaMA)
- Vector search using ChromaDB
- Clean and modern Streamlit UI

## 🛠 Tech Stack
- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Ollama (Local LLM)

## ▶️ How to Run

```bash
pip install -r requirements.txt
ollama serve
ollama pull gemma3:4b
streamlit run app.py
