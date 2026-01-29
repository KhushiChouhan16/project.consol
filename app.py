import streamlit as st
import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="Intelligent PDF Chatbot (RAG)",
    layout="wide"
)

st.title("📄 Intelligent PDF Chatbot (RAG)")
st.caption("Ask questions strictly from the uploaded PDF")

# ================= SESSION STATE =================
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# ================= LAYOUT =================
left_col, right_col = st.columns([1, 2])

# ================= LEFT PANEL (UPLOAD) =================
with left_col:
    st.subheader("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:

        # Reset state ONLY if new PDF uploaded
        if st.session_state.current_pdf != uploaded_file.name:
            st.session_state.chat_history = []
            st.session_state.qa_chain = None
            st.session_state.current_pdf = uploaded_file.name

        VECTOR_DIR = f"vectorstore_{uuid.uuid4().hex}"
        os.makedirs(VECTOR_DIR, exist_ok=True)
        os.makedirs("uploads", exist_ok=True)

        pdf_path = os.path.join("uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("✅ PDF uploaded successfully")
        st.write(f"📄 **{uploaded_file.name}**")

        # -------- LOAD PDF --------
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.info(f"Pages loaded: {len(documents)}")

        # -------- SPLIT TEXT --------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        st.info(f"Chunks created: {len(chunks)}")

        if len(chunks) == 0:
            st.error("❌ No readable text found in this PDF.")
            st.stop()

        # -------- EMBEDDINGS + VECTOR DB --------
        with st.spinner("Creating vector database..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=VECTOR_DIR
            )

        st.success("✅ Vector database ready")

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # -------- LLM (OLLAMA) --------
        llm = Ollama(model="gemma3:4b")

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

# ================= RIGHT PANEL (CHAT) =================
with right_col:
    st.subheader("💬 Chat with your PDF")

    if st.session_state.qa_chain is None:
        st.info("Upload a PDF to start chatting.")
    else:
        # ---------- FIXED QUERY BAR (TOP) ----------
        user_question = st.chat_input("Ask a question from the PDF")

        # ---------- HANDLE QUESTION ----------
        if user_question:
            with st.spinner("Searching document..."):
                answer = st.session_state.qa_chain.run(user_question)

            # Insert latest Q&A at TOP
            st.session_state.chat_history.insert(
                0, (user_question, answer)
            )

        # ---------- SHOW LATEST ANSWER ----------
        if st.session_state.chat_history:
            latest_q, latest_a = st.session_state.chat_history[0]

            with st.chat_message("user"):
                st.markdown(latest_q)

            with st.chat_message("assistant"):
                st.markdown(latest_a)

        # ---------- SHOW OLD HISTORY ----------
        if len(st.session_state.chat_history) > 1:
            st.divider()
            st.markdown("### 📜 Previous Questions")

            for q, a in st.session_state.chat_history[1:]:
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    st.markdown(a)

# ================= FOOTER =================
st.divider()
st.caption("Built with LangChain • ChromaDB • Ollama • Streamlit")
st.caption("Project by Khushi Chouhan")
