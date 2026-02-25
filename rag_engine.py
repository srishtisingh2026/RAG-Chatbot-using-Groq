# rag_engine.py

import time
import tiktoken
import logging
import hashlib
import uuid

logger = logging.getLogger("rag_logger")

class RAGEngine:

    def __init__(self, llm, vector_store, k=2, max_context_tokens=6000, distance_threshold=1.1):
        self.llm = llm
        self.vector_store = vector_store
        self.k = k
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = max_context_tokens
        self.distance_threshold = distance_threshold

    def classify_intent(self, query):
        """Classifies the user query into predefined intents."""
        # Fast heuristic for greetings
        lower_query = query.lower().strip()
        greetings = {"hi", "hello", "hii", "hey", "thanks", "thank you", "good morning", "good afternoon"}
        if lower_query in greetings:
            return "greeting"

        prompt = f"""
        You are an AI intent classifier.
        Label the query as exactly one of the following words:
        - "greeting": hi, hello, thanks, etc.
        - "knowledge_query": factual or procedural questions found in documents.
        - "unrelated": topics outside useful document context.

        Query: "{query}"

        Return ONLY the label (e.g., "greeting"). Extract the intent strictly.
        """
        response = self.llm.invoke(prompt)
        intent_text = response.content.strip().lower()
        
        # Extract first word strictly
        first_word = intent_text.split()[0].replace('"', '').replace('.', '').replace(',', '')
        
        valid_intents = {"greeting", "knowledge_query", "unrelated"}
        if first_word in valid_intents:
            return first_word
        
        # Fallback substring check
        for v in valid_intents:
            if v in intent_text:
                return v
                
        return "unrelated"

    def retrieve_documents(self, query):
        """Retrieves and filters documents based on relevance threshold."""
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.k)
        
        # Filter by threshold
        filtered_docs = []
        for doc, score in docs_with_scores:
            if score <= self.distance_threshold:
                filtered_docs.append((doc, score))
        
        # Truncate to token limit
        truncated_docs = []
        current_tokens = 0
        for doc, score in filtered_docs:
            doc_tokens = len(self.enc.encode(doc.page_content))
            if current_tokens + doc_tokens <= self.max_context_tokens:
                truncated_docs.append((doc, score))
                current_tokens += doc_tokens
            else:
                break
        
        return truncated_docs, docs_with_scores

    def generate_response(self, query, context, intent):
        """Generates the LLM response based on intent and context."""
        if intent == "greeting":
            prompt = f"User: '{query}'. Respond with a short, friendly AI greeting. Under 15 words."
        elif intent == "unrelated":
            prompt = f"User: '{query}'. Explain politely that you can only help with questions related to the document base. Under 20 words."
        else: # knowledge_query
            prompt = f"""
            You are a helpful knowledge assistant. Provide a concise, 1-2 sentence answer.
            Use the context below to answer. If context is empty, say you don't have that information.

            Context:
            {context}

            Question:
            {query}
            """
        
        response = self.llm.invoke(prompt)
        return response.content, prompt, response.response_metadata if hasattr(response, 'response_metadata') else {}

    def run(self, query):
        spans = []
        start_time_ms = int(time.time() * 1000)
        
        # 1. Intent Classification
        intent_start = int(time.time() * 1000)
        intent = self.classify_intent(query)
        intent_end = int(time.time() * 1000)
        
        spans.append({
            "span_id": f"span-intent-{uuid.uuid4().hex[:8]}",
            "type": "intent-classification",
            "name": "intent-classifier",
            "start_time": intent_start,
            "end_time": intent_end,
            "latency_ms": intent_end - intent_start,
            "metadata": {"intent": intent}
        })

        # 2. Retrieval Routing
        retrieval_executed = False
        docs_found = 0
        best_score = 2.0
        safe_docs = []
        
        if intent == "knowledge_query":
            retrieval_executed = True
            retrieval_start = int(time.time() * 1000)
            safe_docs, all_docs = self.retrieve_documents(query)
            retrieval_end = int(time.time() * 1000)
            
            docs_found = len(safe_docs)
            if all_docs:
                best_score = float(all_docs[0][1])
            
            spans.append({
                "span_id": f"span-retrieval-{uuid.uuid4().hex[:8]}",
                "type": "retrieval",
                "name": "vector-search",
                "start_time": retrieval_start,
                "end_time": retrieval_end,
                "latency_ms": retrieval_end - retrieval_start,
                "metadata": {
                    "documents_found": docs_found,
                    "best_score": round(best_score, 4),
                    "threshold": self.distance_threshold
                }
            })
            
            trace_name = "rag-qa" if docs_found > 0 else "no-context-qa"
        elif intent == "greeting":
            trace_name = "simple-qa"
        else:
            trace_name = "out-of-scope-qa"

        # 3. Generation
        context = "\n\n".join([doc.page_content for doc, _ in safe_docs])
        gen_start = int(time.time() * 1000)
        output_text, used_prompt, provider_raw = self.generate_response(query, context, intent)
        gen_end = int(time.time() * 1000)

        spans.append({
            "span_id": f"span-llm-{uuid.uuid4().hex[:8]}",
            "type": "llm",
            "name": "groq_chat_completion",
            "start_time": gen_start,
            "end_time": gen_end,
            "latency_ms": gen_end - gen_start,
            "metadata": {"temperature": 0.0},
            "usage": provider_raw.get("token_usage", {})
        })

        # 4. Telemetry Normalization
        total_latency = spans[-1]["end_time"] - spans[0]["start_time"]
        
        # Retrieval Confidence (1 / (1 + score))
        retrieval_confidence = round(1 / (1 + best_score), 4) if retrieval_executed else 0.0

        rag_data = {
            "retrieved_documents": [
                {
                    "doc_id": hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
                    "score": round(float(score), 4),
                    "content_preview": doc.page_content[:200]
                } for doc, score in safe_docs
            ],
            "documents_found": docs_found,
            "best_score": round(best_score, 4),
            "retrieval_confidence": retrieval_confidence
        }

        prompt_tokens = len(self.enc.encode(used_prompt))
        completion_tokens = len(self.enc.encode(output_text))

        return {
            "output": output_text,
            "context": context,
            "intent": intent,
            "trace_name": trace_name,
            "latency_ms": total_latency,
            "documents_found": docs_found,
            "retrieval_executed": retrieval_executed,
            "retrieval_confidence": retrieval_confidence,
            "rag_data": rag_data,
            "spans": spans,
            "provider_raw": {
                "id": provider_raw.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": provider_raw.get("model_name", "unknown"),
                "choices": [{"message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}],
                "usage": provider_raw.get("token_usage", {})
            },
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }