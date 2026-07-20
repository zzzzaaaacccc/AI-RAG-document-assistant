import streamlit as st
import PyPDF2
import io
import os
import re
import time
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Agentic RAG Document Assistant")

st.title("Agentic RAG Document Assistant")
st.markdown(
    "Upload one or more PDFs. The agent will retrieve, rerank, and if the "
    "context isn't enough, refine its search and retrieve again before answering."
)

MAX_HOPS = 3          # max retrieval iterations the agent can take
CANDIDATES_PER_HOP = 8  # chunks pulled from FAISS before reranking
TOP_N_AFTER_RERANK = 4  # chunks kept after reranking, per hop

CACHE_DIR = ".rag_cache"  # per-file FAISS indexes live here, keyed by content hash
os.makedirs(CACHE_DIR, exist_ok=True)


#  Cached resources

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_reranker():
    # Cross-encoder: scores (query, passage) pairs directly, which is far more
    # accurate than raw embedding similarity for ranking retrieved chunks.
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_llm():
    return OllamaLLM(model="mistral", base_url="http://localhost:11434")


# Session state 

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "messages" not in st.session_state:
    st.session_state.messages = []


# File-level caching helpers 

def hash_bytes(data: bytes) -> str:
    """Content hash used as the cache key, so re-uploading the same file
    (even under a different name) skips PDF extraction and embedding."""
    return hashlib.sha256(data).hexdigest()[:16]


def cache_path_for(file_hash: str) -> str:
    return os.path.join(CACHE_DIR, file_hash)


def load_cached_index(file_hash: str, embeddings):
    path = cache_path_for(file_hash)
    if os.path.isdir(path):
        try:
            return FAISS.load_local(
                path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            return None  # corrupt/incompatible cache — fall through and rebuild
    return None


def save_index_to_cache(index, file_hash: str):
    index.save_local(cache_path_for(file_hash))


def extract_and_chunk(file_bytes: bytes, filename: str, splitter):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    chunks = splitter.split_text(text)
    docs = [
        Document(page_content=chunk, metadata={"source": filename})
        for chunk in chunks
    ]
    return docs, len(pdf_reader.pages)


def process_file(uploaded_file, splitter, embeddings, status_placeholder):
    """Process a single uploaded file, using the on-disk cache when possible.
    Returns (faiss_index_for_this_file, num_pages, was_cached)."""
    file_bytes = uploaded_file.read()
    file_hash = hash_bytes(file_bytes)

    status_placeholder.markdown(f"Checking cache for **{uploaded_file.name}**...")
    cached_index = load_cached_index(file_hash, embeddings)
    if cached_index is not None:
        status_placeholder.markdown(f"**{uploaded_file.name}** loaded from cache (skipped re-embedding)")
        # We still need page count for the summary; re-read cheaply.
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return cached_index, len(pdf_reader.pages), True

    status_placeholder.markdown(f"Extracting text from **{uploaded_file.name}**...")
    docs, num_pages = extract_and_chunk(file_bytes, uploaded_file.name, splitter)

    status_placeholder.markdown(
        f"Embedding {len(docs)} chunk(s) from **{uploaded_file.name}**..."
    )
    file_index = FAISS.from_documents(docs, embeddings)

    status_placeholder.markdown(f"Caching index for **{uploaded_file.name}**...")
    save_index_to_cache(file_index, file_hash)

    return file_index, num_pages, False


# Sidebar: upload 

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embeddings = get_embeddings()

        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        combined_index = None
        doc_names = []
        total_pages = 0
        cached_count = 0

        try:
            for i, uploaded_file in enumerate(uploaded_files):
                file_index, num_pages, was_cached = process_file(
                    uploaded_file, splitter, embeddings, status_placeholder
                )

                if combined_index is None:
                    combined_index = file_index
                else:
                    combined_index.merge_from(file_index)

                doc_names.append(uploaded_file.name)
                total_pages += num_pages
                cached_count += 1 if was_cached else 0

                progress_bar.progress((i + 1) / len(uploaded_files))

            st.session_state.vector_store = combined_index
            st.session_state.doc_names = doc_names
            st.session_state.messages = []

            status_placeholder.empty()
            cache_note = f" ({cached_count} from cache)" if cached_count else ""
            st.success(
                f"Loaded {len(doc_names)} document(s), {total_pages} total pages{cache_note}"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if st.session_state.doc_names:
        st.markdown("**Loaded documents:**")
        for name in st.session_state.doc_names:
            st.markdown(f"- {name}")

        if st.button("Clear all documents"):
            st.session_state.vector_store = None
            st.session_state.doc_names = []
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear disk cache"):
            import shutil
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            st.success("Cache cleared — next upload will re-embed from scratch.")


# Agent logic 

def retrieve_and_rerank(vector_store, query, reranker, k=CANDIDATES_PER_HOP, top_n=TOP_N_AFTER_RERANK):
    """Retrieve candidates with FAISS, then rerank with a cross-encoder."""
    candidates = vector_store.similarity_search(query, k=k)
    if not candidates:
        return []

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_n]]


def retrieve_with_document_coverage(vector_store, query, reranker, doc_names, per_doc_k=2, per_doc_candidates=4):
    """
    Plain similarity search only returns the globally highest-scoring chunks,
    which can silently skip entire documents on broad/comparison questions
    (e.g. "summarize all documents") since no single chunk from a quieter
    document ever wins the top-k race.

    This guarantees at least `per_doc_k` chunks from EVERY loaded document by
    filtering retrieval to one source at a time, then reranking within each
    document's candidates before combining.
    """
    covered_docs = []
    for name in doc_names:
        try:
            candidates = vector_store.similarity_search(
                query, k=per_doc_candidates, filter={"source": name}
            )
        except Exception:
            candidates = []  # filter unsupported or no matches — skip gracefully

        if not candidates:
            continue

        if len(candidates) > per_doc_k:
            pairs = [(query, doc.page_content) for doc in candidates]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            candidates = [doc for doc, _ in ranked[:per_doc_k]]

        covered_docs.extend(candidates)

    return covered_docs


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


def run_agent(vector_store, llm, reranker, question, doc_names, max_hops=MAX_HOPS):
    """
    Multi-hop agentic retrieval loop:
      1. Retrieve + rerank chunks for the current query (hop 1 also guarantees
         coverage across every loaded document, so broad "summarize/compare
         all documents" questions don't silently miss quieter files)
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
        t0 = time.perf_counter()
        docs = retrieve_and_rerank(vector_store, current_query, reranker)

        if hop == 1 and len(doc_names) > 1:
            coverage_docs = retrieve_with_document_coverage(
                vector_store, current_query, reranker, doc_names
            )
            docs = docs + coverage_docs

        retrieval_secs = time.perf_counter() - t0

        new_docs = [d for d in docs if d.page_content not in seen_content]
        for d in new_docs:
            seen_content.add(d.page_content)
        accumulated_docs.extend(new_docs)

        coverage_note = ""
        if hop == 1 and len(doc_names) > 1:
            sources_hit = sorted({d.metadata.get("source", "unknown") for d in accumulated_docs})
            coverage_note = f" (covering {len(sources_hit)}/{len(doc_names)} document(s))"

        trace.append(
            f"**Hop {hop}:** searched for _\"{current_query}\"_ -> "
            f"retrieved {len(docs)} candidates, reranked, kept {len(new_docs)} new chunk(s)"
            f"{coverage_note} "
            f"({retrieval_secs:.2f}s)"
        )

        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in accumulated_docs
        )

        if hop == max_hops:
            trace.append(f"**Hop {hop}:** reached max hops — proceeding to final answer")
            break

        t0 = time.perf_counter()
        decision, refined_query = assess_context(llm, question, context)
        assess_secs = time.perf_counter() - t0

        if decision == "READY":
            trace.append(
                f"**Hop {hop}:** agent judged context sufficient — answering now "
                f"({assess_secs:.2f}s, LLM call)"
            )
            break
        else:
            trace.append(
                f"**Hop {hop}:** agent judged context insufficient — refining search "
                f"({assess_secs:.2f}s, LLM call)"
            )
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

    t0 = time.perf_counter()
    response = llm.invoke(answer_prompt)
    answer_secs = time.perf_counter() - t0
    answer = response.content if hasattr(response, "content") else str(response)

    trace.append(f"**Final answer generation:** ({answer_secs:.2f}s, LLM call)")

    sources = [
        {"name": d.metadata.get("source", "unknown"), "content": d.page_content}
        for d in accumulated_docs
    ]

    return answer, trace, sources


# Main chat 

if st.session_state.vector_store:
    st.subheader(f"{len(st.session_state.doc_names)} document(s) loaded")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "trace" in msg:
                with st.expander("Agent reasoning trace"):
                    for step in msg["trace"]:
                        st.markdown(step)
            if "sources" in msg:
                with st.expander("Sources"):
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

                    t_start = time.perf_counter()
                    answer, trace, sources = run_agent(
                        st.session_state.vector_store, llm, reranker, prompt,
                        st.session_state.doc_names
                    )
                    total_secs = time.perf_counter() - t_start

                    st.markdown(answer)
                    st.caption(f"Total time: {total_secs:.1f}s")

                    with st.expander("Agent reasoning trace"):
                        for step in trace:
                            st.markdown(step)

                    with st.expander("Sources"):
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
    st.info("Upload one or more PDFs to start")
