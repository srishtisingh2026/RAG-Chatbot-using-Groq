# app.py

import streamlit as st
import uuid
import time
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_engine import RAGEngine
from observability import Telemetry


# ---------------------------------------------------------
# Load ENV
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Session Management (6-digit IDs)
# ---------------------------------------------------------

if "session_id" not in st.session_state:

    short_id = uuid.uuid4().hex[:6]

    st.session_state["session_id"] = f"session-{short_id}"
    st.session_state["user_id"] = f"user-{short_id}"
    st.session_state["request_count"] = 0

session_id = st.session_state["session_id"]
user_id = st.session_state["user_id"]


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=selected_model,
    temperature=0.25,
    max_tokens=300
)


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings = get_embeddings()


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

@st.cache_resource
def get_vector_store(_embeddings, last_updated):

    return FAISS.load_local(
        "faiss_index",
        _embeddings,
        allow_dangerous_deserialization=True
    )


if os.path.exists("faiss_index/index.faiss"):

    index_mtime = os.path.getmtime("faiss_index/index.faiss")

    from datetime import datetime

    readable_mtime = datetime.fromtimestamp(index_mtime).strftime("%Y-%m-%d %H:%M:%S")

    st.sidebar.info(f"Vector Index Last Updated: {readable_mtime}")

else:

    index_mtime = 0
    st.sidebar.warning("Vector index not found. Run vectordb.py first.")


vector_store = get_vector_store(embeddings, index_mtime)


# ---------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------

@st.cache_resource
def get_rag_engine(model_name, _llm, _vector_store):

    return RAGEngine(_llm, _vector_store, k=3, distance_threshold=0.3)

rag_engine = get_rag_engine(selected_model, llm, vector_store)


# ---------------------------------------------------------
# Telemetry
# ---------------------------------------------------------

@st.cache_resource
def get_telemetry(url):
    return Telemetry(url)

telemetry = get_telemetry(backend_url)


# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------

with st.form("query_form", clear_on_submit=False):
    query = st.text_input(
        "Ask a question about your documents",
        key="query_input"
    )
    submitted = st.form_submit_button("Get Answer")


# ---------------------------------------------------------
# Run Query
# ---------------------------------------------------------

if submitted and query:

    # Persist the submitted query so it stays visible after rerun
    st.session_state["last_query"] = query
    st.session_state["request_count"] += 1

    trace_id = f"trace-{uuid.uuid4().hex[:8]}"

    # Show the question clearly above the spinner
    st.markdown(f"**🧑 You asked:** {query}")

    with st.spinner("🔍 Searching documents and generating answer..."):

        start_time_ms = int(time.time() * 1000)

        result = rag_engine.run(query)



    # ---------------------------------------------------------
    # Safe Defaults (prevent KeyErrors)
    # ---------------------------------------------------------

    documents_found = result.get("documents_found", 0)
    retrieval_executed = result.get("retrieval_executed", False)
    retrieval_confidence = result.get("retrieval_confidence", 0)
    rag_data = result.get("rag_data", {})
    spans = result.get("spans", [])
    provider_raw = result.get("provider_raw", {})
    trace_name = result.get("trace_name", "unknown")
    intent = result.get("intent", "unknown")
    latency = result.get("latency_ms", 0)


    # ---------------------------------------------------------
    # Trace Log
    # ---------------------------------------------------------

    log_data = {

        "id": trace_id,
        "partitionKey": trace_id,

        "trace_id": trace_id,
        "trace_name": trace_name,

        "session_id": session_id,
        "user_id": user_id,

        "timestamp": start_time_ms,
        "environment": "dev",

        "intent": intent,
        "routing_decision": result.get("routing_decision"),

        "provider": "groq",
        "model": selected_model,

        "input": {
            "query": query
        },

        "output": {
            "answer": result.get("output", "")
        },

        "latency_ms": latency,

        "documents_found": documents_found,
        "retrieval_executed": retrieval_executed,
        "retrieval_confidence": retrieval_confidence,

        "rag_data": rag_data,
        "spans": spans,

        "provider_raw": provider_raw,

        "status": "success"
    }

    telemetry.log_trace(log_data)


    # ---------------------------------------------------------
    # Display Output
    # ---------------------------------------------------------

    st.markdown(f"**🤖 Answer:**")
    st.write(result.get("output", "No response generated"))

# Show last question if no new query is running
elif st.session_state.get("last_query"):
    st.info(f"Last question: {st.session_state['last_query']}")