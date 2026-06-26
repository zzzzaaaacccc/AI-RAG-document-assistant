import streamlit as st
import PyPDF2
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")

st.title("📄 RAG Document Assistant")
st.markdown("Upload a PDF and ask questions about it")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Upload
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and st.button("📤 Process Document"):
        with st.spinner("Processing..."):
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                chunks = splitter.split_text(text)
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []
                
                st.success(f"✅ Loaded '{uploaded_file.name}' ({len(pdf_reader.pages)} pages)")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Main chat
if st.session_state.vector_store:
    st.subheader(f"📖 {st.session_state.doc_name}")
    
    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📌 Sources"):
                    for source in msg["sources"]:
                        st.text(source[:300])
    
    # Chat input
    if prompt := st.chat_input("Ask about the document..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                    sources = [doc.page_content for doc in docs]
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    llm = Ollama(model="mistral", base_url="http://localhost:11434")
                    
                    prompt_text = f"""Answer based ONLY on context:

Context:
{context}

Question: {prompt}

Answer:"""
                    
                    response = llm.invoke(prompt_text)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    with st.expander("📌 Sources"):
                        for i, source in enumerate(sources, 1):
                            st.text(f"Source {i}: {source[:300]}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Upload a PDF to start")