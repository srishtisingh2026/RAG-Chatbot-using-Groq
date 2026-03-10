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
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="RAG Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Advanced CSS for State-of-the-Art Aesthetic
# ---------------------------------------------------------

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background: #f8f9fa !important;
    }

    /* Soft premium background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #ffffff 0%, #f1f3f5 100%) !important;
        color: #212529;
    }

    /* Remove padding around the main area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* Sidebar - Light & Minimal */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(0, 0, 0, 0.05);
    }

    /* Abstract Header */
    .header-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2.5rem;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 24px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
    }

    .main-title {
        font-weight: 800;
        font-size: 3.5rem;
        letter-spacing: -2px;
        margin: 0;
        background: linear-gradient(to right, #212529 30%, #4361ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        font-weight: 400;
        font-size: 0.8rem;
        color: rgba(0, 0, 0, 0.4);
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 1rem;
    }

    /* Chat Bubbles - Elevated Style */
    div.stChatMessage {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid rgba(0, 0, 0, 0.03);
        border-radius: 20px;
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
    }

    /* User specific - Subtle Blue */
    [data-testid="stChatMessageUser"] {
        background: rgba(67, 97, 238, 0.05) !important;
        border: 1px solid rgba(67, 97, 238, 0.1) !important;
    }

    /* Chat Input */
    .stChatInputContainer {
        border: none !important;
        background: transparent !important;
    }

    .stChatInputContainer input {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        border-radius: 16px !important;
        color: #212529 !important;
        padding: 1rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
    }

    /* Sidebar info cards */
    .sidebar-card {
        background: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.03);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    /* Hide redundant elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #dee2e6; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #ced4da; }
    </style>
""", unsafe_allow_html=True)

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

# Title Section
st.markdown('<h1 style="font-weight: 800; font-size: 3rem; letter-spacing: -1.5px; color: #212529;">RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 3rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("### SETTINGS")
    selected_model = st.selectbox("MODEL", AVAILABLE_MODELS)


# ---------------------------------------------------------
# Session Management (Silent)
# ---------------------------------------------------------

if "session_id" not in st.session_state:
    short_id = uuid.uuid4().hex[:6]
    st.session_state["session_id"] = f"session-{short_id}"
    st.session_state["user_id"] = f"user-{short_id}"

session_id = st.session_state["session_id"]
user_id = st.session_state["user_id"]


# ---------------------------------------------------------
# LLM & Embeddings initialization (cached)
# ---------------------------------------------------------

@st.cache_resource
def init_llm(model_name, api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.25,
        max_tokens=300
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

llm = init_llm(selected_model, groq_api_key)
embeddings = get_embeddings()


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

@st.cache_resource
def get_vector_store(_embeddings, last_updated):
    if not os.path.exists("faiss_index/index.faiss"):
        return None
    return FAISS.load_local(
        "faiss_index",
        _embeddings,
        allow_dangerous_deserialization=True
    )

# Get index update time for caching logic
index_mtime = os.path.getmtime("faiss_index/index.faiss") if os.path.exists("faiss_index/index.faiss") else 0
vector_store = get_vector_store(embeddings, index_mtime)


# ---------------------------------------------------------
# RAG Engine & Telemetry
# ---------------------------------------------------------

@st.cache_resource
def get_rag_engine(model_name, _llm, _vector_store):
    if not _vector_store:
        return None
    return RAGEngine(_llm, _vector_store, k=3, distance_threshold=0.3)

@st.cache_resource
def get_telemetry(url):
    return Telemetry(url)

rag_engine = get_rag_engine(selected_model, llm, vector_store)
telemetry = get_telemetry(backend_url)


# ---------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------

# Display history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 1. Chat Input
if prompt := st.chat_input("Send prompt..."):

    # 2. Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Assistant response
    if not rag_engine:
        with st.chat_message("assistant"):
            st.error("Engine unavailable. Index missing.")
    else:
        with st.chat_message("assistant"):
            
            trace_id = f"trace-{uuid.uuid4().hex[:8]}"
            
            with st.spinner("Processing..."):
                start_time_ms = int(time.time() * 1000)
                result = rag_engine.run(prompt)
                
                raw_answer = result.get("output", "Empty signal.")
                
                # --- ULTIMATE MULTI-STAGE DEDUPLICATOR ---
                import re
                
                # Stage 1: Paragraph Deduplication
                paragraphs = [p.strip() for p in raw_answer.split('\n') if p.strip()]
                unique_paragraphs = []
                seen_paras = set()
                for p in paragraphs:
                    p_key = "".join(re.sub(r'[^a-zA-Z0-9]', '', p)).lower()
                    if p_key not in seen_paras:
                        unique_paragraphs.append(p)
                        seen_paras.add(p_key)
                
                # Stage 2: Sentence/Segment Deduplication
                temp_text = " ".join(unique_paragraphs)
                segments = re.split(r'(?<=[.!?])\s*', temp_text)
                final_segments = []
                seen_segments = set()
                for s in segments:
                    s_clean = s.strip()
                    if not s_clean: continue
                    s_key = "".join(re.sub(r'[^a-zA-Z0-9]', '', s_clean)).lower()
                    if s_key not in seen_segments:
                        final_segments.append(s_clean)
                        seen_segments.add(s_key)
                
                answer = " ".join(final_segments).strip()
                if not answer and raw_answer:
                    answer = raw_answer.strip()

                latency = result.get("latency_ms", 0)
                docs = result.get("documents_found", 0)
                
                # Telemetry
                telemetry.log_trace({
                    "id": trace_id, "partitionKey": trace_id, "trace_id": trace_id,
                    "session_id": session_id, "user_id": user_id,
                    "timestamp": start_time_ms,
                    "input": {"query": prompt}, "output": {"answer": answer},
                    "latency_ms": latency, "documents_found": docs,
                    "status": "success"
                })

            # Display and store
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer
            })