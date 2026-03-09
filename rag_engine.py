import time
import math
import tiktoken
import logging
import hashlib
import uuid
import statistics
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag_logger")


class RAGEngine:

    def __init__(self, llm, vector_store, k=6, max_context_tokens=900, distance_threshold=0.5):

        self.llm = llm
        self.vector_store = vector_store
        self.k = k
        self.max_context_tokens = max_context_tokens
        self.distance_threshold = distance_threshold

        self.enc = tiktoken.get_encoding("cl100k_base")

        print("Initializing BM25 index...")

        all_docs = []
        if hasattr(self.vector_store, "docstore"):
            all_docs = list(self.vector_store.docstore._dict.values())

        if all_docs:
            self.bm25 = BM25Retriever.from_documents(all_docs)
            self.bm25.k = 8
        else:
            self.bm25 = None

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )

    # ---------------------------------------------------------
    # Intent Classification
    # ---------------------------------------------------------

    def classify_intent(self, query):

        greetings = {
            "hi", "hello", "hey",
            "thanks", "thank you",
            "good morning", "good afternoon"
        }

        if query.lower().strip() in greetings:
            return "greeting", {}

        prompt = f"""
Classify the intent.

Options:
- greeting
- search_query

Query: "{query}"

Return only label.
"""

        response = self.llm.invoke(prompt)

        usage = response.response_metadata.get(
            "token_usage", {}
        ) if hasattr(response, "response_metadata") else {}

        if "greeting" in response.content.lower():
            return "greeting", usage

        return "search_query", usage

    # ---------------------------------------------------------
    # Query Rewrite
    # ---------------------------------------------------------

    def rewrite_query(self, query):

        prompt = f"""
Rewrite this query to improve document retrieval.

Query:
{query}

Return only the rewritten query.
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()

    # ---------------------------------------------------------
    # Reciprocal Rank Fusion
    # ---------------------------------------------------------

    def reciprocal_rank_fusion(self, vector_docs, bm25_docs, k=60):

        scores = {}

        for rank, (doc, _) in enumerate(vector_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)

        merged = []

        seen = {}

        for doc, _ in vector_docs:
            seen[doc.page_content] = doc

        for doc in bm25_docs:
            seen[doc.page_content] = doc

        for content, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            merged.append(seen[content])

        return merged

    # ---------------------------------------------------------
    # MMR Diversity Filter
    # ---------------------------------------------------------

    def mmr_filter(self, docs, query, lambda_param=0.7):

        selected = []
        seen = set()

        for doc in docs:

            text = doc.page_content[:200]

            if text in seen:
                continue

            seen.add(text)
            selected.append(doc)

            if len(selected) >= self.k * 3:
                break

        return selected

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    def retrieve_documents(self, query):

        vector_docs = self.vector_store.similarity_search_with_score(
            query,
            k=self.k * 4
        )

        bm25_docs = []
        if self.bm25:
            bm25_docs = self.bm25.get_relevant_documents(query)

        fused_docs = self.reciprocal_rank_fusion(vector_docs, bm25_docs)

        diversified_docs = self.mmr_filter(fused_docs, query)

        if diversified_docs:

            pairs = [[query, doc.page_content] for doc in diversified_docs]

            rerank_scores = self.reranker.predict(pairs)

            normalized_scores = [
                1 / (1 + math.exp(-s))
                for s in rerank_scores
            ]

            docs_with_scores = sorted(
                zip(diversified_docs, normalized_scores),
                key=lambda x: x[1],
                reverse=True
            )

        else:
            docs_with_scores = []

        truncated_docs = []

        current_tokens = 0
        seen_parents = set()

        for doc, score in docs_with_scores:

            if score < self.distance_threshold:
                continue

            content = doc.metadata.get(
                "parent_context",
                doc.page_content
            )

            if content in seen_parents:
                continue

            tokens = len(self.enc.encode(content))

            if tokens > self.max_context_tokens:
                content = doc.page_content
                tokens = len(self.enc.encode(content))

            if current_tokens + tokens <= self.max_context_tokens:

                doc.page_content = content

                truncated_docs.append((doc, score))

                current_tokens += tokens

                seen_parents.add(content)

            if len(truncated_docs) >= self.k:
                break

        return truncated_docs, docs_with_scores

    # ---------------------------------------------------------
    # Generation
    # ---------------------------------------------------------

    def generate_response(self, query, context, intent, routing_decision):

        if intent == "greeting":

            prompt = f"""
User: {query}

Respond with a short friendly greeting.
"""

        elif routing_decision == "out_of_scope":

            prompt = f"""
User: {query}

Explain politely that you only answer questions about the provided documents.
"""

        else:

            prompt = f"""
You are a knowledge assistant.

Answer the question using the context below.

If the answer can be reasonably inferred from the context, answer it.

If the context clearly does not contain the answer say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{query}

Maximum 2 sentences.
"""

        response = self.llm.invoke(prompt)

        metadata = response.response_metadata if hasattr(
            response, "response_metadata") else {}

        return response.content, prompt, metadata

    # ---------------------------------------------------------
    # Main Pipeline
    # ---------------------------------------------------------

    def run(self, query):

        spans = []

        start = int(time.time() * 1000)
        intent_start = start

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

        routing_decision = None
        trace_name = "simple-qa"

        safe_docs = []

        retrieval_executed = intent == "search_query"

        if intent == "search_query":

            rewritten = self.rewrite_query(query)

            retrieval_start = int(time.time() * 1000)

            docs1, _ = self.retrieve_documents(query)
            docs2, _ = self.retrieve_documents(rewritten)

            retrieval_end = int(time.time() * 1000)

            combined = docs1 + docs2

            unique = {}

            for doc, score in combined:
                key = doc.page_content
                if key not in unique or score > unique[key][1]:
                    unique[key] = (doc, score)

            safe_docs = sorted(
                unique.values(),
                key=lambda x: x[1],
                reverse=True
            )[:self.k]

            best_score = max(
                (float(score) for _, score in safe_docs),
                default=None
            )

            spans.append({
                "span_id": f"span-retrieval-{uuid.uuid4().hex[:8]}",
                "type": "retrieval",
                "name": "vector-search",
                "start_time": retrieval_start,
                "end_time": retrieval_end,
                "latency_ms": retrieval_end - retrieval_start,
                "metadata": {
                    "documents_found": len(safe_docs),
                    "best_score": round(best_score, 4) if best_score else None,
                    "threshold": self.distance_threshold
                }
            })

            if best_score and best_score >= self.distance_threshold:
                routing_decision = "rag"
                trace_name = "rag-qa"
            else:
                routing_decision = "out_of_scope"
                trace_name = "out-of-scope-qa"

        context = ""

        if routing_decision == "rag":

            context = "\n\n".join(
                f"SOURCE {i+1}:\n{doc.page_content}"
                for i, (doc, _) in enumerate(safe_docs)
            )

        llm_start = int(time.time() * 1000)

        output, used_prompt, provider_raw = self.generate_response(
            query,
            context,
            intent,
            routing_decision
        )

        llm_end = int(time.time() * 1000)
        llm_usage = provider_raw.get("token_usage", {})
        context_tokens = len(self.enc.encode(context)) if context else 0
        spans.append({
            "span_id": f"span-llm-{uuid.uuid4().hex[:8]}",
            "type": "llm",
            "name": "groq_chat_completion",
            "start_time": llm_start,
            "end_time": llm_end,
            "latency_ms": llm_end - llm_start,
            "metadata": {
                "temperature": 0.25,
                "context_tokens": context_tokens
            },
            "usage": llm_usage
        })

        scores = [float(score) for _, score in safe_docs]

        if scores:

            avg_score = statistics.mean(scores)

            coverage = min(len(scores) / 3, 1)

            std_score = statistics.pstdev(scores) if len(scores) > 1 else 0

            consistency = 1 / (1 + std_score)

            retrieval_confidence = (
                0.6 * avg_score +
                0.25 * coverage +
                0.15 * consistency
            )

        else:
            retrieval_confidence = 0

        latency = int(time.time() * 1000) - start

        return {

            "output": output,
            "context": context,

            "intent": intent,
            "routing_decision": routing_decision,
            "trace_name": trace_name,

            "latency_ms": latency,

            "retrieval_executed": retrieval_executed,
            "documents_found": len(safe_docs),
            "retrieval_confidence": round(retrieval_confidence, 4),

            "spans": spans,
            "provider_raw": provider_raw,

            "rag_data": {

                "documents_found": len(safe_docs),

                "retrieval_scores": {
                    "avg_score": round(avg_score, 4) if scores else 0,
                    "min_score": round(min(scores), 4) if scores else 0,
                    "max_score": round(max(scores), 4) if scores else 0,
                    "std_score": round(std_score, 4) if len(scores) > 1 else 0,
                    "per_doc_scores": [round(s, 4) for s in scores]
                },

                "retrieved_documents": [

                    {
                        "doc_id": hashlib.md5(
                            doc.page_content.encode()
                        ).hexdigest()[:8],

                        "score": round(float(score), 4),

                        "content_preview": doc.page_content
                    }

                    for doc, score in safe_docs
                ]
            }
        }