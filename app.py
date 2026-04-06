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
import smartllmops
import vectordb
import pandas as pd
import httpx
import groq
from langchain_community.document_loaders import PyPDFLoader

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

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b"
]

# Title Section
st.markdown('<h1 style="font-weight: 800; font-size: 3rem; letter-spacing: -1.5px; color: #212529;">RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 3rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar - Early selection for model
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
    # Create custom httpx clients with longer keepalive
    sync_httpx = httpx.Client(
        limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=300)
    )
    async_httpx = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=300)
    )
    
    # Manually initialize groq clients to bypass langchain-groq validation bug
    client = groq.Groq(api_key=api_key, http_client=sync_httpx).chat.completions
    async_client = groq.AsyncGroq(api_key=api_key, http_client=async_httpx).chat.completions

    return ChatGroq(
        model_name=model_name,
        temperature=0.25,
        max_tokens=300,
        client=client,
        async_client=async_client
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
def get_rag_engine_v2(model_name, _llm, _vector_store, _tracer):
    if not _vector_store:
        return None
    return RAGEngine(_llm, _vector_store, tracer=_tracer, k=6, distance_threshold=0.50)

@st.cache_resource
def get_sdk_tracer_v2(env, salt="final_reboot_v11"):
    return smartllmops.init(
        environment=env
    )

sdk_tracer = get_sdk_tracer_v2("dev")
rag_engine = get_rag_engine_v2(selected_model, llm, vector_store, sdk_tracer)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------

with st.sidebar:
    
    st.markdown("---")
    st.markdown("### UPLOAD DOCUMENTS")
    uploaded_files = st.file_uploader("Upload PDF, TXT or CSV", type=["pdf", "txt", "csv"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process & Index Files"):
            with st.spinner("Indexing new files..."):
                new_docs = []
                for uploaded_file in uploaded_files:
                    # Save to Data directory
                    if not os.path.exists("Data"):
                        os.makedirs("Data")
                    
                    file_path = os.path.join("Data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text based on type
                    content = ""
                    if uploaded_file.name.endswith(".pdf"):
                        # We use PyPDFLoader for consistency with vectordb.py
                        loader = PyPDFLoader(file_path)
                        pdf_pages = loader.load()
                        content = "\n".join([p.page_content for p in pdf_pages])
                    elif uploaded_file.name.endswith(".txt"):
                        content = uploaded_file.read().decode("utf-8")
                    elif uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                        content = df.astype(str).agg(" ".join, axis=1).str.cat(sep="\n")
                    
                    if content:
                        chunks = vectordb.chunk_text(content, uploaded_file.name)
                        new_docs.extend(chunks)
                
                if new_docs:
                    # Load existing or create new
                    if vector_store:
                        vector_store.add_documents(new_docs)
                    else:
                        # Initialize new index
                        emb_model = vectordb.get_embedding_model()
                        vector_store = FAISS.from_documents(
                            new_docs, 
                            emb_model, 
                            distance_strategy=vectordb.DistanceStrategy.MAX_INNER_PRODUCT
                        )
                    
                    # Save updated index
                    vector_store.save_local("faiss_index")
                    st.success(f"Indexed {len(new_docs)} chunks!")
                    st.info("Click 'Reload Engine' to apply changes.")
    
    st.markdown("---")
    if st.button("Sync Database"):
        with st.spinner("Rebuilding index from Data folder..."):
            vectordb.build_vector_store()
            st.success("Database synced! Click 'Reload Engine' to apply.")

    if st.button("Reload Engine"):
        st.cache_resource.clear()
        st.rerun()




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
            with st.spinner("Processing..."):
                start_time_ms = int(time.time() * 1000)
                sdk_tracer.start_trace()
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

                # Telemetry via SDK (Self-Assembly Mode)
                sdk_tracer.export_trace(
                    result, 
                    query=prompt, 
                    session_id=session_id, 
                    user_id=user_id, 
                    timestamp=start_time_ms,
                    rag_docs=result.get("safe_docs")
                )

            # Display and store
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer
            })