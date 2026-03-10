# vectordb.py

import os
import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

DATA_FOLDER = "Data"
INDEX_FOLDER = "faiss_index"


# ---------------------------------------------------------
# Utilities for Chunking & Indexing
# ---------------------------------------------------------

def get_splitters():
    # Parent chunks (larger context for generation)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Child chunks (precise retrieval)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return parent_splitter, child_splitter

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def chunk_text(raw_text, filename):
    parent_splitter, child_splitter = get_splitters()
    source_label = os.path.splitext(filename)[0]
    
    final_documents = []
    parent_chunks = parent_splitter.split_text(raw_text)

    for p_idx, parent_text in enumerate(parent_chunks):
        child_chunks = child_splitter.split_text(parent_text)
        for c_idx, child_text in enumerate(child_chunks):
            enriched_content = f"[Source: {source_label}]\n{child_text}"
            doc = Document(
                page_content=enriched_content,
                metadata={
                    "source": filename,
                    "parent_context": f"[Source: {source_label}]\n{parent_text}",
                    "chunk_id": f"{filename}_p{p_idx}_c{c_idx}"
                }
            )
            final_documents.append(doc)
    return final_documents

# ---------------------------------------------------------
# Document Loader
# ---------------------------------------------------------

def load_documents():
    documents = []
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        
    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)
        try:
            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                documents.append((text, file))
            elif file.endswith(".csv"):
                df = pd.read_csv(file_path)
                text = df.astype(str).agg(" ".join, axis=1).str.cat(sep="\n")
                documents.append((text, file))
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                text = "\n".join([doc.page_content for doc in pdf_docs])
                documents.append((text, file))
        except Exception as e:
            print(f"Failed to load {file}: {e}")
    return documents

# ---------------------------------------------------------
# Build Vector Store
# ---------------------------------------------------------

def build_vector_store():
    print("Loading documents...")
    raw_data = load_documents()
    
    all_chunks = []
    for raw_text, filename in raw_data:
        all_chunks.extend(chunk_text(raw_text, filename))

    print(f"Total chunks created: {len(all_chunks)}")
    
    if not all_chunks:
        print("No documents to index.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Building FAISS index...")
    vector_store = FAISS.from_documents(
        all_chunks,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )

    if not os.path.exists(INDEX_FOLDER):
        os.makedirs(INDEX_FOLDER)

    vector_store.save_local(INDEX_FOLDER)
    print(f"Vector DB created successfully in {INDEX_FOLDER}")

if __name__ == "__main__":
    import streamlit as st # Only needed for cache deco if run as main, but better to import at top
    build_vector_store()