import streamlit as st
import PyPDF2
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")

st.title("📄 RAG Document Assistant")
st.markdown("Upload one or more PDFs and ask questions across all of them")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Upload
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📤 Process Documents"):
        with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )

                all_docs = []
                doc_names = []
                total_pages = 0

                for uploaded_file in uploaded_files:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""

                    chunks = splitter.split_text(text)

                    # Tag every chunk with its source filename so answers
                    # can cite which document they came from
                    for chunk in chunks:
                        all_docs.append(
                            Document(
                                page_content=chunk,
                                metadata={"source": uploaded_file.name},
                            )
                        )

                    doc_names.append(uploaded_file.name)
                    total_pages += len(pdf_reader.pages)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vector_store = FAISS.from_documents(all_docs, embeddings)

                st.session_state.vector_store = vector_store
                st.session_state.doc_names = doc_names
                st.session_state.messages = []

                st.success(
                    f"✅ Loaded {len(doc_names)} document(s), "
                    f"{total_pages} total pages"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.doc_names:
        st.markdown("**Loaded documents:**")
        for name in st.session_state.doc_names:
            st.markdown(f"- {name}")

        if st.button("🗑️ Clear all documents"):
            st.session_state.vector_store = None
            st.session_state.doc_names = []
            st.session_state.messages = []
            st.rerun()

# Main chat
if st.session_state.vector_store:
    st.subheader(f"📖 {len(st.session_state.doc_names)} document(s) loaded")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📌 Sources"):
                    for source in msg["sources"]:
                        st.text(f"[{source['name']}] {source['content'][:300]}")

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                    sources = [
                        {
                            "name": doc.metadata.get("source", "unknown"),
                            "content": doc.page_content,
                        }
                        for doc in docs
                    ]
                    context = "\n\n".join(
                        f"[Source: {s['name']}]\n{s['content']}" for s in sources
                    )

                    llm = Ollama(model="mistral", base_url="http://localhost:11434")

                    prompt_text = f"""Answer based ONLY on the context below. If the context comes from multiple documents, mention which document(s) support your answer.

Context:
{context}

Question: {prompt}

Answer:"""

                    response = llm.invoke(prompt_text)
                    answer = response.content if hasattr(response, "content") else str(response)

                    st.markdown(answer)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                    with st.expander("📌 Sources"):
                        for i, source in enumerate(sources, 1):
                            st.text(f"Source {i} [{source['name']}]: {source['content'][:300]}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Upload one or more PDFs to start")
