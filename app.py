import streamlit as st
import PyPDF2
import io
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Agentic RAG Document Assistant", page_icon="📄")

st.title("📄 Agentic RAG Document Assistant")
st.markdown(
    "Upload one or more PDFs. The agent will retrieve, rerank, and — if the "
    "context isn't enough — refine its search and retrieve again before answering."
)

MAX_HOPS = 3          # max retrieval iterations the agent can take
CANDIDATES_PER_HOP = 8  # chunks pulled from FAISS before reranking
TOP_N_AFTER_RERANK = 4  # chunks kept after reranking, per hop


# ---------- Cached resources ----------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_reranker():
    # Cross-encoder: scores (query, passage) pairs directly, which is far more
    # accurate than raw embedding similarity for ranking retrieved chunks.
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_llm():
    return Ollama(model="mistral", base_url="http://localhost:11434")


# ---------- Session state ----------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Sidebar: upload ----------

with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("📤 Process Documents"):
        with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
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
                    for chunk in chunks:
                        all_docs.append(
                            Document(
                                page_content=chunk,
                                metadata={"source": uploaded_file.name},
                            )
                        )

                    doc_names.append(uploaded_file.name)
                    total_pages += len(pdf_reader.pages)

                embeddings = get_embeddings()
                vector_store = FAISS.from_documents(all_docs, embeddings)

                st.session_state.vector_store = vector_store
                st.session_state.doc_names = doc_names
                st.session_state.messages = []

                st.success(
                    f"✅ Loaded {len(doc_names)} document(s), {total_pages} total pages"
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


# ---------- Agent logic ----------

def retrieve_and_rerank(vector_store, query, reranker, k=CANDIDATES_PER_HOP, top_n=TOP_N_AFTER_RERANK):
    """Retrieve candidates with FAISS, then rerank with a cross-encoder."""
    candidates = vector_store.similarity_search(query, k=k)
    if not candidates:
        return []

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_n]]


def assess_context(llm, question, context):
    """
    Ask the LLM whether the current context is sufficient to answer.
    Returns ('READY', None) or ('NEED_MORE', refined_query).
    This is the agentic decision step: the model decides the next action.
    """
    prompt = f"""You are assessing whether you have enough information to answer a question.

Context so far:
{context}

Question: {question}

If the context is sufficient to fully answer the question, respond with exactly:
READY

If the context is NOT sufficient, respond with exactly:
NEED_MORE: <a refined, more specific search query that would help find the missing information>

Respond with only one of the two formats above, nothing else."""

    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    text = text.strip()

    match = re.search(r"NEED_MORE:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return "NEED_MORE", match.group(1).strip().split("\n")[0]
    return "READY", None


def run_agent(vector_store, llm, reranker, question, max_hops=MAX_HOPS):
    """
    Multi-hop agentic retrieval loop:
      1. Retrieve + rerank chunks for the current query
      2. Ask the LLM if it has enough context
      3. If not, refine the query and retrieve again (up to max_hops)
      4. Synthesize a final answer from all accumulated context
    Returns (answer, trace, sources) where trace is a list of step descriptions
    for transparency in the UI.
    """
    trace = []
    accumulated_docs = []
    seen_content = set()
    current_query = question

    for hop in range(1, max_hops + 1):
        docs = retrieve_and_rerank(vector_store, current_query, reranker)

        new_docs = [d for d in docs if d.page_content not in seen_content]
        for d in new_docs:
            seen_content.add(d.page_content)
        accumulated_docs.extend(new_docs)

        trace.append(
            f"**Hop {hop}:** searched for _\"{current_query}\"_ → "
            f"retrieved {len(docs)} candidates, reranked, kept {len(new_docs)} new chunk(s)"
        )

        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in accumulated_docs
        )

        if hop == max_hops:
            trace.append(f"**Hop {hop}:** reached max hops — proceeding to final answer")
            break

        decision, refined_query = assess_context(llm, question, context)

        if decision == "READY":
            trace.append(f"**Hop {hop}:** agent judged context sufficient → answering now")
            break
        else:
            trace.append(f"**Hop {hop}:** agent judged context insufficient → refining search")
            current_query = refined_query

    # Final answer synthesis
    final_context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in accumulated_docs
    )

    answer_prompt = f"""Answer the question based ONLY on the context below. Cite which document(s) support your answer.

Context:
{final_context}

Question: {question}

Answer:"""

    response = llm.invoke(answer_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    sources = [
        {"name": d.metadata.get("source", "unknown"), "content": d.page_content}
        for d in accumulated_docs
    ]

    return answer, trace, sources


# ---------- Main chat ----------

if st.session_state.vector_store:
    st.subheader(f"📖 {len(st.session_state.doc_names)} document(s) loaded")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "trace" in msg:
                with st.expander("🧠 Agent reasoning trace"):
                    for step in msg["trace"]:
                        st.markdown(step)
            if "sources" in msg:
                with st.expander("📌 Sources"):
                    for source in msg["sources"]:
                        st.text(f"[{source['name']}] {source['content'][:300]}")

    if prompt := st.chat_input("Ask about your documents..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Agent is retrieving and reasoning..."):
                try:
                    llm = get_llm()
                    reranker = get_reranker()

                    answer, trace, sources = run_agent(
                        st.session_state.vector_store, llm, reranker, prompt
                    )

                    st.markdown(answer)

                    with st.expander("🧠 Agent reasoning trace"):
                        for step in trace:
                            st.markdown(step)

                    with st.expander("📌 Sources"):
                        for i, source in enumerate(sources, 1):
                            st.text(f"Source {i} [{source['name']}]: {source['content'][:300]}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "trace": trace,
                            "sources": sources,
                        }
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Upload one or more PDFs to start")
