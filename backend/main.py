from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import PyPDF2
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import traceback

load_dotenv()

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
documents_store = {}
retriever = None
llm = None

class UploadResponse(BaseModel):
    message: str
    doc_id: str
    pages: int

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF"""
    global retriever, llm
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    try:
        content = await file.read()
        
        # Extract text
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        page_count = len(pdf_reader.pages)
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_text(text)
        
        # Create documents for retriever
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        # Use BM25 retriever (keyword-based, no embeddings needed)
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 3
        
        # Initialize LLM with gemini-2.0-flash
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
        doc_id = file.filename.replace('.pdf', '').replace(' ', '_')
        documents_store[doc_id] = {
            'name': file.filename,
            'pages': page_count,
            'chunks': len(chunks)
        }
        
        return UploadResponse(
            message=f"Successfully processed {file.filename}",
            doc_id=doc_id,
            pages=page_count
        )
    
    except Exception as e:
        print(f"Upload error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_document(req: QueryRequest):
    """Query the uploaded document"""
    global retriever, llm
    
    if retriever is None or llm is None:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")
    
    try:
        # Get relevant documents
        docs = retriever.invoke(req.question)
        sources = [doc.page_content[:200] for doc in docs]
        
        # Build context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create the prompt
        prompt_template = """You are a helpful assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Format and invoke
        formatted_prompt = prompt.format(context=context, question=req.question)
        response = llm.invoke(formatted_prompt)
        
        # Extract text from response
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        print(f"Query error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List uploaded documents"""
    return documents_store

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)