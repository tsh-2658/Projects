import csv
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

# Constants
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "vectorstore"
LOG_CSV = BASE_DIR / "upload_logs.csv"
DB_FILE = BASE_DIR / "app.db"
LOG_TEXT = BASE_DIR / "upload_logs.txt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# Database Setup
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
metadata = MetaData()

upload_logs = Table(
    "upload_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(128), nullable=False),
    Column("filename", String(256), nullable=False),
    Column("document_name", String(256), nullable=False),
    Column("upload_time", DateTime, nullable=False),
    Column("source", String(256), nullable=True),
    Column("notes", Text, nullable=True),
)

metadata.create_all(engine)

# In-memory storage
chat_memories: Dict[str, List[Dict[str, str]]] = defaultdict(list)

app = FastAPI(title="GenAI RAG Portal")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# GOOGLE Config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = "gemini-2.0-flash" 

if ChatGoogleGenerativeAI is None:
    raise RuntimeError("ChatGOOGLE model is required.")

llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GOOGLE_MODEL, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store: Optional[FAISS] = None

class ChatRequest(BaseModel):
    question: str

# --- Helper Functions (Logs) ---
def read_log_csv() -> List[Dict[str, Any]]:
    if not LOG_CSV.exists(): return []
    with LOG_CSV.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def save_log_csv(rows: List[Dict[str, Any]]):
    with LOG_CSV.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "username", "filename", "document_name", "upload_time"])
        writer.writeheader()
        writer.writerows(rows)

def dump_logs_to_text():
    rows = read_log_csv()
    lines = [f"{row['upload_time']} | {row['username']} | {row['document_name']} | {row['filename']}" for row in rows]
    LOG_TEXT.write_text("\n".join(lines), encoding="utf-8")

def is_log_question(question: str) -> bool:
    keywords = ["upload", "uploaded", "log", "recent uploads", "last 7 days"]
    return any(keyword in question.lower() for keyword in keywords)

def answer_log_question(question: str) -> Optional[str]:
    if not is_log_question(question): return None
    # ... (Keep existing log parsing logic)
    now = datetime.utcnow()
    query = select(upload_logs).order_by(upload_logs.c.upload_time.desc())
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
    if not rows: return "No uploads found."
    entries = [f"{row.upload_time.isoformat()} - {row.filename}" for row in rows]
    return "Upload Log Details:\n" + "\n".join(entries)

def create_db_log(username: str, filename: str, document_name: str):
    now = datetime.utcnow()
    with engine.begin() as conn:
        conn.execute(upload_logs.insert().values(username=username, filename=filename, document_name=document_name, upload_time=now))
    rows = read_log_csv()
    rows.append({"id": str(len(rows) + 1), "username": username, "filename": filename, "document_name": document_name, "upload_time": now.isoformat()})
    save_log_csv(rows)
    dump_logs_to_text()

# --- Summarization Function ---
def summarize_all_documents() -> str:
    """Summarizes all files in the upload directory using Map-Reduce."""
    docs: List[Document] = []
    for file_path in sorted(UPLOAD_DIR.iterdir()):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs.extend(loader.load())
            elif ext in {".txt", ".md", ".csv"}:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs.extend(loader.load())
    
    if not docs:
        return "No documents found to summarize."

    # Split for summarization (slightly larger chunks for better context)
    summary_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = summary_splitter.split_documents(docs)

    # Use Map-Reduce to handle many documents (100+)
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    return summarize_chain.run(split_docs)

# --- RAG Logic ---
def build_vector_store() -> Optional[FAISS]:
    docs: List[Document] = []
    for file_path in sorted(UPLOAD_DIR.iterdir()):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs.extend(loader.load_and_split())
            elif ext in {".txt", ".md", ".csv"}:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs.extend(chunker.split_documents(loader.load()))
    if not docs: return None
    index = FAISS.from_documents(docs, embeddings)
    index.save_local(str(VECTOR_DIR))
    return index

def get_vector_store() -> Optional[FAISS]:
    global vector_store
    if vector_store: return vector_store
    if (VECTOR_DIR / "index.faiss").exists():
        vector_store = FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
        return vector_store
    vector_store = build_vector_store()
    return vector_store

def rag_search(question: str, username: str) -> str:
    store = get_vector_store()
    if not store: return "No documents found."
    retriever = store.as_retriever(search_kwargs={"k": 4})
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the context:
    Context: {context}
    Question: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": question})
    return result.get("answer", "")

# --- API Routes ---
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    global vector_store
    for upload in files:
        target_path = UPLOAD_DIR / upload.filename
        with target_path.open("wb") as buffer:
            buffer.write(await upload.read())
        create_db_log("admin_user", upload.filename, upload.filename)
    vector_store = build_vector_store()
    return {"message": "Upload complete"}

@app.post("/api/chat")
def chat(payload: ChatRequest):
    username = "admin_user"
    question = payload.question.lower()

    # 1. Log Info Check
    log_answer = answer_log_question(question)
    if log_answer:
        return {"question": payload.question, "answer": log_answer}

    # 2. Summarization Check
    summary_keywords = ["summarize all", "summary of all", "summarize everything", "overall summary"]
    if any(k in question for k in summary_keywords):
        summary = summarize_all_documents()
        return {"question": payload.question, "answer": summary}

    # 3. Normal RAG Chat
    answer = rag_search(payload.question, username)
    return {"question": payload.question, "answer": answer}

@app.get("/", response_class=HTMLResponse)
def read_index():
    return HTMLResponse((BASE_DIR / "static" / "index.html").read_text(encoding="utf-8"))

@app.get("/api/logs")
def get_logs():
    with engine.connect() as conn:
        result = conn.execute(select(upload_logs).order_by(upload_logs.c.upload_time.desc()))
        return {"logs": [dict(row._mapping) for row in result.fetchall()]}