# app.py

import streamlit as st
import uuid
import hashlib
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_engine import RAGEngine
from observability import Telemetry
import time

# Load ENV
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
backend_url = os.getenv("BACKEND_TRACE_URL")

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b"
]

st.title("Groq RAG Chatbot")

selected_model = st.selectbox("Select Model", AVAILABLE_MODELS)

@st.cache_resource
def get_session_counter():
    return {"count": 0}

session_counter = get_session_counter()

if "session_id" not in st.session_state:
    session_counter["count"] += 1
    session_idx = session_counter["count"]
    st.session_state["session_id"] = f"session-{session_idx:02d}"
    st.session_state["user_id"] = f"user-{session_idx:02d}"
    st.session_state["request_count"] = 0

session_id = st.session_state["session_id"]
user_id = st.session_state["user_id"]

# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=selected_model,
    temperature=0.3,
    max_tokens=200
)

# Vector DB
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def get_vector_store(_embeddings, last_updated):
    return FAISS.load_local(
        "faiss_index",
        _embeddings,
        allow_dangerous_deserialization=True
    )

embeddings = get_embeddings()

# Cache busting based on folder modification time
if os.path.exists("faiss_index"):
    index_mtime = os.path.getmtime("faiss_index")
    import datetime
    readable_mtime = datetime.datetime.fromtimestamp(index_mtime).strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.info(f"Vector Index Last Updated: {readable_mtime}")
else:
    index_mtime = 0
    st.sidebar.warning("Vector index not found. Please run vectordb.py")

vector_store = get_vector_store(embeddings, index_mtime)

# RAG Engine
rag_engine = RAGEngine(llm, vector_store, k=2)
telemetry = Telemetry(backend_url)

query = st.text_input("Ask a question about your documents")

if st.button("Get Answer") and query:

    st.session_state["request_count"] += 1
    trace_id = f"trace-{st.session_state['request_count']:02d}"

    # Run RAG Engine with spinner
    with st.spinner("Searching and generating answer..."):
        start_time_ms = int(time.time() * 1000)
        result = rag_engine.run(query)

    # Final Log Format
    log_data = {
        "trace_id": trace_id,
        "trace_name": result["trace_name"],
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": start_time_ms,
        "environment": "dev",
        "intent": result["intent"],
        "provider": "groq",
        "model": selected_model,
        "input": {
            "query": query
        },
        "output": {
            "answer": result["output"]
        },
        "latency_ms": result["latency_ms"],
        "documents_found": result["documents_found"],
        "retrieval_executed": result["retrieval_executed"],
        "retrieval_confidence": result["retrieval_confidence"],
        "rag_data": result["rag_data"],
        "spans": result["spans"],
        "provider_raw": result["provider_raw"],
        "status": "success"
    }

    # Log trace
    telemetry.log_trace(log_data)

    st.write(result["output"])
    st.markdown(f"Total Tokens: {result['usage']['total_tokens']}")
