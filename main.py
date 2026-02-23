import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Loading the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# -----------------------------
# Tool 1: Wikipedia Tool
# -----------------------------
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=10000
)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# -----------------------------
# Cached Resources
# -----------------------------

@st.cache_resource
def get_vector_db():
    logger.info("Initializing HuggingFace embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"device": "cpu"}
        )
        
        logger.info("Loading documents from directory...")
        loader = PyPDFDirectoryLoader("./us_census_data")
        docs = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        documents = text_splitter.split_documents(docs)

        logger.info(f"Building FAISS vector database from {len(documents)} document chunks...")
        vectordb = FAISS.from_documents(documents, embeddings)
        return vectordb
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}", exc_info=True)
        st.error(f"Initialization error: {e}")
        return None

# Initialize resources
vectordb = get_vector_db()

if vectordb:
    # Retriever
    retriever = vectordb.as_retriever()
else:
    st.error("Vector database could not be initialized. PDF search will not be available.")
    retriever = None

from langchain.tools.retriever import create_retriever_tool
# -----------------------------
# Tool 3: Arxiv Tool
# -----------------------------
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=10000
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

if retriever:
    pdf_tool = create_retriever_tool(
        retriever,
        "pdf_search",
        "Search for information about US census data. For any questions about US census data, you must use this tool first!"
    )
else:
    pdf_tool = None

# Filter out None tools
tools = [wiki, arxiv]
if pdf_tool:
    tools.append(pdf_tool)

# -----------------------------
# Streamlit Setup
# -----------------------------
st.title("Chatbot using Groq")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the user's questions as accurately as possible.
For questions about US census data, you MUST use the 'pdf_search' tool.
For general knowledge, you can use 'wikipedia_search'.
For scientific queries, use 'arxiv_search'.

If context is provided from a tool, use it. If no context is found but you have tools, mention that you're checking.

Questions: {input}
{agent_scratchpad}
""")

# Agent Setup
from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Agent Executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# -----------------------------
# Frontend Input + Execution
# -----------------------------
query = st.text_input("Input your query here")

if st.button("Get Answer") or query:
    logger.info(f"Received query: {query}")
    start_overall = time.time()
    start_llm = time.process_time()

    try:
        response = agent_executor.invoke({
            "input": query,
            "agent_scratchpad": ""
        })

        response_time_overall = time.time() - start_overall
        response_time_llm = time.process_time() - start_llm

        logger.info(f"Response generated in {response_time_overall:.2f}s. Input: {query}")
        st.write(response['output'])

        st.markdown(
            f"<p style='color:blue;'>Overall Response Time: {response_time_overall:.4f} seconds</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color:blue;'>LLM Response Time: {response_time_llm:.4f} seconds</p>",
            unsafe_allow_html=True
        )

    except Exception as e:
        logger.error(f"Error processing query '{query}': {str(e)}", exc_info=True)
        st.write(f"An error occurred: {e}")

else:
    st.write("Please enter a query")