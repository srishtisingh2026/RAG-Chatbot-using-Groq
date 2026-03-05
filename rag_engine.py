import time
import tiktoken
import logging
import hashlib
import uuid
import statistics

logger = logging.getLogger("rag_logger")


class RAGEngine:

    def __init__(self, llm, vector_store, k=2, max_context_tokens=6000, distance_threshold=0.6):
        self.llm = llm
        self.vector_store = vector_store
        self.k = k
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = max_context_tokens
        self.distance_threshold = distance_threshold

    def classify_intent(self, query):
        """Classifies the user query into basic categories."""

        lower_query = query.lower().strip()
        greetings = {"hi", "hello", "hii", "hey", "thanks", "thank you", "good morning", "good afternoon"}

        if lower_query in greetings:
            return "greeting", {}

        prompt = f"""
You are an AI intent classifier.

Label the query as exactly one of the following:
- greeting
- search_query

Query: "{query}"

Return ONLY the label.
"""

        response = self.llm.invoke(prompt)

        intent_text = response.content.strip().lower()
        usage = response.response_metadata.get("token_usage", {})

        if "greeting" in intent_text:
            return "greeting", usage

        return "search_query", usage

    def retrieve_documents(self, query_vector):
        """Retrieve documents and apply threshold + token truncation."""

        docs_with_scores = self.vector_store.similarity_search_with_score_by_vector(
            query_vector,
            k=self.k
        )

        truncated_docs = []
        current_tokens = 0

        for doc, score in docs_with_scores:
            if score >= self.distance_threshold:
                doc_tokens = len(self.enc.encode(doc.page_content))
                if current_tokens + doc_tokens <= self.max_context_tokens:
                    truncated_docs.append((doc, score))
                    current_tokens += doc_tokens
                else:
                    break

        return truncated_docs, docs_with_scores

    def generate_response(self, query, context, intent, routing_decision=None):
        """Generate response using LLM."""

        if intent == "greeting":

            prompt = f"User: '{query}'. Respond with a short friendly greeting under 15 words."

        elif routing_decision == "out_of_scope":

            prompt = f"User: '{query}'. Politely explain you only answer questions about the document base. Under 20 words."

        else:

            prompt = f"""
You are a helpful knowledge assistant.

Provide a concise 1-2 sentence answer.

Use the context below.

If the context is empty say you don't have that information.

Context:
{context}

Question:
{query}
"""

        response = self.llm.invoke(prompt)

        return (
            response.content,
            prompt,
            response.response_metadata if hasattr(response, "response_metadata") else {}
        )

    def run(self, query):

        spans = []

        start_time_ms = int(time.time() * 1000)

        # -----------------------------
        # 1 Intent Classification
        # -----------------------------

        intent_start = int(time.time() * 1000)

        intent, intent_usage = self.classify_intent(query)

        intent_end = int(time.time() * 1000)

        spans.append({
            "span_id": f"span-intent-{uuid.uuid4().hex[:8]}",
            "type": "intent-classification",
            "name": "intent-classifier",
            "start_time": intent_start,
            "end_time": intent_end,
            "latency_ms": intent_end - intent_start,
            "metadata": {"intent": intent},
            "usage": intent_usage
        })

        retrieval_executed = False
        routing_decision = None
        docs_found = 0
        safe_docs = []
        all_docs = []

        # -----------------------------
        # 2 Retrieval
        # -----------------------------

        if intent == "search_query":

            retrieval_executed = True

            emb_start = int(time.time() * 1000)

            query_vector = self.vector_store.embeddings.embed_query(query)

            emb_end = int(time.time() * 1000)

            spans.append({
                "span_id": f"span-embedding-{uuid.uuid4().hex[:8]}",
                "type": "embedding",
                "name": "query-embedding",
                "start_time": emb_start,
                "end_time": emb_end,
                "latency_ms": emb_end - emb_start,
                "metadata": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
            })

            retrieval_start = int(time.time() * 1000)

            safe_docs, all_docs = self.retrieve_documents(query_vector)

            retrieval_end = int(time.time() * 1000)

            docs_found = len(safe_docs)

            best_score = None

            if all_docs:
                best_score = max(float(score) for _, score in all_docs)

            spans.append({
                "span_id": f"span-retrieval-{uuid.uuid4().hex[:8]}",
                "type": "retrieval",
                "name": "vector-search",
                "start_time": retrieval_start,
                "end_time": retrieval_end,
                "latency_ms": retrieval_end - retrieval_start,
                "metadata": {
                    "documents_found": docs_found,
                    "best_score": round(best_score, 4) if best_score else None,
                    "threshold": self.distance_threshold
                }
            })

            if docs_found > 0:
                routing_decision = "rag"
                trace_name = "rag-qa"
            else:
                routing_decision = "out_of_scope"
                trace_name = "out-of-scope-qa"

        elif intent == "greeting":

            trace_name = "simple-qa"

        else:

            trace_name = "unknown-qa"

        # -----------------------------
        # 3 Generation
        # -----------------------------

        context = "\n\n".join([doc.page_content for doc, _ in safe_docs])

        gen_start = int(time.time() * 1000)

        output_text, used_prompt, provider_raw = self.generate_response(
            query,
            context,
            intent,
            routing_decision
        )

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

        # -----------------------------
        # 4 Retrieval Score Statistics
        # -----------------------------

        scores = [float(score) for _, score in all_docs[:self.k]]

        if scores:

            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            std_score = statistics.pstdev(scores) if len(scores) > 1 else 0

            retrieval_confidence = round(avg_score, 4)

        else:

            avg_score = None
            min_score = None
            max_score = None
            std_score = None
            retrieval_confidence = 0.0

        rag_data = {

            "retrieved_documents": [
                {
                    "doc_id": hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
                    "score": round(float(score), 4),
                    "content_preview": doc.page_content[:200]
                }
                for doc, score in safe_docs
            ],

            "documents_found": docs_found,

            "retrieval_scores": {
                "avg_score": round(avg_score, 4) if avg_score is not None else None,
                "min_score": round(min_score, 4) if min_score is not None else None,
                "max_score": round(max_score, 4) if max_score is not None else None,
                "std_score": round(std_score, 4) if std_score is not None else None,
                "per_doc_scores": [round(float(s), 4) for s in scores]
            },

            "retrieval_confidence": retrieval_confidence
        }

        # -----------------------------
        # Usage Aggregation
        # -----------------------------

        total_prompt_tokens = intent_usage.get("prompt_tokens", 0) + \
            provider_raw.get("token_usage", {}).get("prompt_tokens", 0)

        total_completion_tokens = intent_usage.get("completion_tokens", 0) + \
            provider_raw.get("token_usage", {}).get("completion_tokens", 0)

        total_latency = spans[-1]["end_time"] - spans[0]["start_time"]

        # -----------------------------
        # Final Trace Object
        # -----------------------------

        return {

            "output": output_text,
            "context": context,

            "intent": intent,
            "routing_decision": routing_decision,
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
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": output_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": provider_raw.get("token_usage", {})
            },

            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }