from embedder import Embedder
from vector_store import VectorStore
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any


class Retriever:
    def __init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.reranker = CrossEncoder(reranker_model)

    def process_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed(query)

        candidate_count = max(top_k * 3, 10)
        retrieved_chunks = self.vector_store.search(query_embedding, candidate_count)

        if not retrieved_chunks:
            return []

        reranked_chunks = self.rerank(query, retrieved_chunks)

        return reranked_chunks[:top_k]

    def rerank(self, query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        """
        Score and re-order chunks based on fine-grained token match with the query.
        """
        if not retrieved_chunks:
            return []

        pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]

        scores = self.reranker.predict(pairs)

        for chunk, score in zip(retrieved_chunks, scores):
            chunk["rerank_score"] = float(score)

        retrieved_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        return retrieved_chunks
