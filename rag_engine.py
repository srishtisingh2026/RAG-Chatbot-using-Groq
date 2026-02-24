# rag_engine.py

import time
import tiktoken
import logging

logger = logging.getLogger("rag_logger")

class RAGEngine:

    def __init__(self, llm, retriever, max_context_tokens=6000):
        self.llm = llm
        self.retriever = retriever
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = max_context_tokens

    def truncate_context(self, retrieved_docs):
        """Truncates the context to stay within the token limit."""
        truncated_docs = []
        current_tokens = 0
        
        for doc in retrieved_docs:
            doc_tokens = len(self.enc.encode(doc.page_content))
            if current_tokens + doc_tokens <= self.max_context_tokens:
                truncated_docs.append(doc)
                current_tokens += doc_tokens
            else:
                logger.warning(f"Truncating context: reached limit of {self.max_context_tokens} tokens.")
                break
        
        return truncated_docs

    def run(self, query, model_name):

        spans = []
        skip_rag = len(query.split()) <= 1

        if skip_rag:
            # ---------------- Skip Retrieval ----------------
            context = ""
            spans.append({
                "name": "vector-search",
                "type": "retrieval",
                "latency_ms": 0.0,
                "status": "skipped"
            })
        else:
            # ---------------- Retrieval ----------------
            retrieval_start = time.time()
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            # Apply truncation safety
            safe_docs = self.truncate_context(retrieved_docs)
            
            retrieval_latency = round((time.time() - retrieval_start) * 1000, 2)

            spans.append({
                "name": "vector-search",
                "type": "retrieval",
                "latency_ms": retrieval_latency
            })

            context = "\n\n".join([doc.page_content for doc in safe_docs])

        # ---------------- Generation ----------------
        generation_start = time.time()

        rag_prompt = f"""
        You are a helpful assistant. Provide a concise, 1-2 sentence answer.
        Do not elaborate unless specifically asked.
        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {query}
        """

        response = self.llm.invoke(rag_prompt)
        generation_latency = round((time.time() - generation_start) * 1000, 2)

        spans.append({
            "name": "generate-response",
            "type": "generation",
            "latency_ms": generation_latency
        })

        output_text = response.content

        # ---------------- Tokens & Cost ----------------
        prompt_tokens = len(self.enc.encode(rag_prompt))
        completion_tokens = len(self.enc.encode(output_text))
        total_tokens = prompt_tokens + completion_tokens


        return {
            "output": output_text,
            "spans": spans,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }