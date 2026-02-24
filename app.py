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
    temperature=0,
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

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

rag_engine = RAGEngine(llm, retriever)
telemetry = Telemetry(backend_url)

query = st.text_input("Ask a question about your documents")

if st.button("Get Answer") and query:

    st.session_state["request_count"] += 1
    request_id = f"request-{st.session_state['request_count']:02d}"

    # Run RAG Engine and track latency
    start_time = time.time()
    result = rag_engine.run(query, selected_model)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Consolidate flat log data
    log_data = {
        "request_id": request_id,
        "session_id": session_id,
        "user_id": user_id,
        "provider": "groq",
        "model": selected_model,
        "input": query,
        "output": result["output"],
        "latency_ms": latency_ms,
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "status": "success"
    }

    # Build and log trace (with spans)
    trace = telemetry.build_trace(log_data, result["spans"])
    telemetry.log_trace(trace)

    st.write(result["output"])
    st.markdown(f"Total Tokens: {result['total_tokens']}")
