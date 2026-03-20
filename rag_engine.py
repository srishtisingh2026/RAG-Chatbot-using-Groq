import time
import math
import tiktoken
import logging
import sys
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import functools

logger = logging.getLogger("rag_logger")


def smart_trace(span_type, name=None):
    """Declarative trace decorator that finds the tracer on the instance (self)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "tracer") and self.tracer:
                span_name = name or func.__name__
                # Dynamic name for LLM spans
                if span_type == "llm" and "{provider}" in span_name:
                    provider = getattr(self.tracer, "provider", "llm")
                    span_name = span_name.format(provider=provider)
                
                return self.tracer.trace(
                    name=span_name, 
                    span_type=span_type,
                    include_io=False
                )(func)(self, *args, **kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class RAGEngine:

    def __init__(self, llm, vector_store, tracer=None, k=6, max_context_tokens=2000, distance_threshold=0.60):
        self.llm = llm
        self.vector_store = vector_store
        self.tracer = tracer
        self.k = k
        self.max_context_tokens = max_context_tokens
        self.distance_threshold = distance_threshold
        
        self.enc = tiktoken.get_encoding("cl100k_base")

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

    @smart_trace(span_type="intent-classification", name="intent-classifier")
    def classify_intent(self, query):

        prompt = f"""
Classify the intent.

Options:
- greeting
- search_query

Query: "{query}"

Return only label.
"""

        response = self.llm.invoke(prompt)

        metadata = getattr(response, "response_metadata", {})
        usage = (
            metadata.get("token_usage")
            or metadata.get("usage_metadata")
            or metadata
        )

        if "greeting" in response.content.lower():
            return "greeting", usage

        return "search_query", usage

    # ---------------------------------------------------------
    # Query Rewrite
    # ---------------------------------------------------------

    @smart_trace(span_type="chain", name="query-rewrite")
    def rewrite_query(self, query):

        prompt = f"""
INSTRUCTION: Rewrite the user's query into a descriptive natural language search phrase to improve vector document retrieval.
RULES:
1. Do NOT generate SQL, code, or structured database queries.
2. Use synonyms and technical variations of the terms.
3. Return ONLY the rewritten natural language phrase.

User Query:
{query}

Rewritten Phrase:
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

    def mmr_filter(self, docs, query):

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

    @smart_trace(span_type="retrieval", name="vector-search")
    def retrieve_documents(self, query):

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
                import torch

                with torch.no_grad():
                    rerank_scores = self.reranker.predict(
                        pairs,
                        batch_size=32,
                        convert_to_tensor=True
                    ).cpu().numpy()

                normalized_scores = [
                    1 / (1 + math.exp(-s))
                    for s in rerank_scores
                ]

            except Exception as e:

                logger.error(f"Reranker failed: {str(e)}")

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

        return truncated_docs, docs_with_scores

    # ---------------------------------------------------------
    # Generation
    # ---------------------------------------------------------

    @smart_trace(span_type="llm", name="{provider}_chat_completion")
    def generate_response(self, query, context, intent, routing_decision):

        if intent == "greeting":

            prompt = f"User: {query}\n\nRespond with a short friendly greeting.\n\nResponse:"

            response = self.llm.invoke(prompt)

            return response.content, prompt, getattr(response, "response_metadata", {})

        if routing_decision == "out_of_scope":

            prompt = f"""
SYSTEM INSTRUCTION: The query is OUT OF SCOPE.

User Query: {query}

Respond with a short refusal (max 1 sentence).
"""

            response = self.llm.invoke(prompt)

            return response.content, prompt, getattr(response, "response_metadata", {})

        prompt = f"""
SYSTEM RULES:
1. Use ONLY the provided context.
2. If answer is not in context say "I don't have enough information".
3. Keep answer under 2 sentences.

Context:
{context}

Question:
{query}

Response:
"""

        response = self.llm.invoke(prompt)

        return response.content, prompt, getattr(response, "response_metadata", {})

    # ---------------------------------------------------------
    # Main Pipeline
    # ---------------------------------------------------------

    @smart_trace(span_type="generic", name="rag-pipeline")
    def run(self, query):

        intent, _ = self.classify_intent(query)

        routing_decision = None
        trace_name = "simple-qa"

        safe_docs = []

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

        output, _, _ = self.generate_response(
            query,
            context,
            intent,
            routing_decision
        )

        return {
            "trace_name": trace_name,
            "output": output,
            "safe_docs": safe_docs,
            "intent": intent,
            "routing_decision": routing_decision
        }