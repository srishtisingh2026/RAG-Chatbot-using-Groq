# vectordb.py

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

DATA_FOLDER = "Data"
INDEX_FOLDER = "faiss_index"


def load_documents():
    documents = []

    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)

        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                documents.append(f.read())

        elif file.endswith(".csv"):
            df = pd.read_csv(file_path)
            documents.append(df.to_string())

        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                documents.append(doc.page_content)

        else:
            print(f"Skipping unsupported file: {file}")

    return documents


def build_vector_store():
    print("Loading documents...")
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    texts = splitter.split_text("\n".join(documents))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(texts, embeddings)
    vector_store.save_local(INDEX_FOLDER)

    print(f"Vector DB created successfully with {len(texts)} chunks.")


if __name__ == "__main__":
    build_vector_store()