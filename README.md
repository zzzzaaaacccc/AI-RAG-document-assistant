# 📄 Agentic RAG Document Assistant

A Streamlit application that lets you upload one or more PDFs and ask natural-language questions about them. Unlike a standard single-pass RAG pipeline, this app uses an **agentic, multi-hop retrieval loop**: the model assesses whether it has enough context to answer, and if not, autonomously refines its search query and retrieves again — up to several hops — before generating a final, source-grounded answer.

## ✨ Features

- **Multi-PDF upload** — upload and query across multiple documents at once
- **Agentic multi-hop retrieval** — the LLM decides after each retrieval step whether it has sufficient context, or whether to generate a refined query and search again (up to 3 hops)
- **Two-stage ranking pipeline** — FAISS embedding similarity search narrows candidates, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks them for higher relevance before they're used as context
- **Visible reasoning trace** — every answer includes an expandable panel showing exactly what the agent searched for at each hop and why it decided to stop or continue
- **Local LLM inference** — answers are generated via Ollama running Mistral locally (no external API calls for generation)
- **Source-aware answers** — every answer cites which document(s) it was derived from, with the exact source passages viewable in an expandable panel
- **Chat-style UI** — persistent conversation history within a session, with an option to clear loaded documents and start fresh

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration / agent loop | LangChain + custom multi-hop controller |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Reranking | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| LLM | Ollama (Mistral) |
| PDF parsing | PyPDF2 |

## 🧠 How the agent works

1. **Retrieve** — the current query is embedded and FAISS returns the top candidate chunks across all loaded documents
2. **Rerank** — a cross-encoder scores each (query, chunk) pair directly for relevance, keeping only the strongest matches
3. **Assess** — the LLM is asked whether the accumulated context is sufficient to answer the original question
4. **Refine or answer** — if context is insufficient, the LLM generates a more targeted follow-up query and the loop repeats (up to 3 hops); otherwise, it proceeds to generate the final answer
5. **Synthesize** — a final answer is generated from all accumulated context, citing which source document(s) support it

## 📸 Screenshots

**Starting page**

![Starting page](starting_page.png)

**Asking a question**

![First response](repsonse.png)

![First response](repsonse_two.png)

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Ollama installed locally (https://ollama.com), with the Mistral model pulled:

```
ollama pull mistral
ollama serve
```

### Installation

```
git clone https://github.com/zzzzaaaacccc/AI-RAG-document-assistant.git
cd AI-RAG-document-assistant
pip install -r requirements.txt
```

### Run

```
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501), upload one or more PDFs from the sidebar, and start asking questions.

## 📂 Project Structure

```
.
├── app.py              # Streamlit app: upload, chunk, embed, agentic retrieval loop, rerank, chat UI
├── requirements.txt
├── screenshots/
│   ├── starting_page.png
│   ├── response.png
│   └── response_part2.png
└── README.md
```

## 🔮 Possible Improvements

- Persist vector stores to disk so documents don't need to be re-processed each session
- Add per-document filtering (query only a subset of loaded files)
- Swap in a hosted LLM option (e.g. Gemini) as a configurable alternative to local Ollama
- Add streaming responses for a more real-time chat feel
- Add evaluation metrics (retrieval precision, answer faithfulness) to quantify agent performance
