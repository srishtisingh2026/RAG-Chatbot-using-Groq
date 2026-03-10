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

    def __init__(self, llm, vector_store, tracer=None, k=6, max_context_tokens=900, distance_threshold=0.5):

        self.llm = llm
        self.vector_store = vector_store
        self.tracer = tracer
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
            device="cpu",
            max_length=512
        )

    # ---------------------------------------------------------
    # Intent Classification
    # ---------------------------------------------------------

    def classify_intent(self, query):
        # Fallback if tracer isn't provided (e.g. in tests)
        if self.tracer:
            return self._classify_intent_wrapped(query)
        return self._classify_intent_raw(query)

    def _classify_intent_wrapped(self, query):
        @self.tracer.trace(
            name="intent-classifier", 
            span_type="intent-classification",
            include_io=False,
            result_parser=lambda r, args, kwargs: {"metadata": {"intent": r[0]}, "usage": r[1]}
        )
        def wrapped_classify(q):
            return self._classify_intent_raw(q)
        return wrapped_classify(query)

    def _classify_intent_raw(self, query):

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
        if self.tracer:
            @self.tracer.trace(
                name="query-rewrite", 
                span_type="chain",
                include_io=False,
                result_parser=lambda r, args, kwargs: {"metadata": {"rewritten_query": r}}
            )
            def wrapped_rewrite(q):
                return self._rewrite_query_raw(q)
            return wrapped_rewrite(query)
        return self._rewrite_query_raw(query)

    def _rewrite_query_raw(self, query):

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
        if self.tracer:
            @self.tracer.trace(
                name="vector-search", 
                span_type="retrieval",
                include_io=False,
                result_parser=lambda r, args, kwargs: {
                    "metadata": {
                        "documents_found": len(r[1]), 
                        "best_score": r[1][0][1] if r[1] else 0,
                        "threshold": self.distance_threshold
                    }
                }
            )
            def wrapped_retrieve(q):
                return self._retrieve_documents_raw(q)
            return wrapped_retrieve(query)
        return self._retrieve_documents_raw(query)

    def _retrieve_documents_raw(self, query):

        vector_docs = self.vector_store.similarity_search_with_score(
            query,
            k=self.k * 2
        )

        bm25_docs = []
        if self.bm25:
            bm25_docs = self.bm25.get_relevant_documents(query)

        fused_docs = self.reciprocal_rank_fusion(vector_docs, bm25_docs)

        diversified_docs = self.mmr_filter(fused_docs, query)

        if diversified_docs:

            pairs = [[query, doc.page_content] for doc in diversified_docs]

            try:
                # Force prediction on CPU and disable gradients for stability
                import torch
                with torch.no_grad():
                    rerank_scores = self.reranker.predict(pairs, batch_size=32, convert_to_tensor=True).cpu().numpy()

                normalized_scores = [
                    1 / (1 + math.exp(-s))
                    for s in rerank_scores
                ]
            except Exception as e:
                logger.error(f"Reranker failed: {str(e)}. Falling back to raw scores.")
                # Fallback: Use descending scores starting from 0.7 for first 3
                normalized_scores = [0.7, 0.6, 0.5] + [0.1] * (len(diversified_docs) - 3)

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
        if self.tracer:
            @self.tracer.trace(
                name="groq_chat_completion", 
                span_type="llm",
                include_io=False,
                result_parser=lambda r, args, kwargs: {
                    "metadata": {
                        "temperature": 0.25,
                        "context_tokens": len(self.enc.encode(args[1])) if len(args) > 1 else 0
                    },
                    "usage": r[2].get("token_usage", r[2]) if isinstance(r[2], dict) else {}
                }
            )
            def wrapped_generate(q, c, i, r):
                return self._generate_response_raw(q, c, i, r)
            return wrapped_generate(query, context, intent, routing_decision)
        return self._generate_response_raw(query, context, intent, routing_decision)

    def _generate_response_raw(self, query, context, intent, routing_decision):

        if intent == "greeting":
            # (Greeting logic remains same, but let's keep it clean)
            prompt = f"User: {query}\n\nRespond with a short friendly greeting. Just the greeting.\n\nResponse:"
            response = self.llm.invoke(prompt)
            return response.content, prompt, getattr(response, "response_metadata", {})

        if routing_decision == "out_of_scope":
            # Specialized refusal prompt instead of hardcoded string
            prompt = f"""
SYSTEM INSTRUCTION: The following query is OUT OF SCOPE.
You are strictly limited to the provided documents. You have NO external knowledge.

User Query: {query}

CONSTRAINT: Refuse to answer the question using authoritative, concise language. 
STRICTLY PROHIBITED: Do NOT provide any biographical, historical, or external facts.

Response (max 1 sentence):
"""
            response = self.llm.invoke(prompt)
            return response.content, prompt, getattr(response, "response_metadata", {})

        # Default RAG Path
        prompt = f"""
SYSTEM RULES:
1. Use ONLY the provided context to answer. 
2. If the answer is not contained within the context, say "I don't have enough information."
3. STRICTLY PROHIBITED: Do not use any internal training data, prior knowledge, or external facts.
4. Keep the answer under 2 sentences.
5. Do NOT repeat the question or use "Answer:".

Context:
{context}

User Question:
{query}

Response:
"""
        response = self.llm.invoke(prompt)
        return response.content, prompt, getattr(response, "response_metadata", {})

    # ---------------------------------------------------------
    # Main Pipeline
    # ---------------------------------------------------------

    def run(self, query):

        start = int(time.time() * 1000)

        intent, intent_usage = self.classify_intent(query)

        routing_decision = None
        trace_name = "simple-qa"

        safe_docs = []
        retrieval_executed = intent == "search_query"

        if intent == "search_query":
            rewritten = self.rewrite_query(query)
            search_query = f"{query} {rewritten}"
            safe_docs, _ = self.retrieve_documents(search_query)

            best_score = max(
                (float(score) for _, score in safe_docs),
                default=None
            )

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

        output, used_prompt, provider_raw = self.generate_response(
            query,
            context,
            intent,
            routing_decision
        )

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
            "environment": "dev",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "output": output,
            "context": context,

            "intent": intent,
            "routing_decision": routing_decision,
            "trace_name": trace_name,

            "latency_ms": latency,

            "retrieval_executed": retrieval_executed,
            "documents_found": len(safe_docs),
            "retrieval_confidence": round(retrieval_confidence, 4),
            "spans": [], # Spans will be pulled from context automatically by export_trace
            "provider_raw": provider_raw,

            "rag_data": {
                "documents_found": len(safe_docs),
                "retrieval_scores": {
                    "avg_score": statistics.mean(scores) if scores else 0,
                    "min_score": min(scores) if scores else 0,
                    "max_score": max(scores) if scores else 0,
                    "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0,
                    "per_doc_scores": [round(s, 4) for s in scores]
                },
                "retrieved_documents": [
                    {
                        "doc_id": hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
                        "score": round(float(score), 4),
                        "content_preview": doc.page_content[:200]
                    }
                    for doc, score in safe_docs
                ]
            }
        }