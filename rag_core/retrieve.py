"""Hybrid retrieval: combine meaning-based and keyword search via RRF."""
from rank_bm25 import BM25Okapi
from .arabic import normalize_ar, stem_tokens
from .store import VectorStore


class HybridRetriever:
    def __init__(self, store: VectorStore, rrf_k: int = 60, use_bm25: bool = True):
        self.store = store
        self.rrf_k = rrf_k
        pairs = store.all_docs()
        self.ids = [p[0] for p in pairs]
        self.docs = [p[1] for p in pairs]
        self.bm25 = (BM25Okapi([stem_tokens(d) for d in self.docs])
                     if use_bm25 and self.docs else None)

    def search(self, query: str, k: int = 5, dense_only: bool = False,
               bm25_only: bool = False) -> list[dict]:
        fused: dict[str, dict] = {}
        if not bm25_only:
            for rank, hit in enumerate(self.store.query(query, k=k * 2)):
                e = fused.setdefault(hit["id"], {"text": hit["text"], "score": 0.0})
                e["score"] += 1.0 / (self.rrf_k + rank + 1)
        if self.bm25 and not dense_only:
            scores = self.bm25.get_scores(stem_tokens(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[: k * 2]
            for rank, i in enumerate(order):
                e = fused.setdefault(self.ids[i], {"text": self.docs[i], "score": 0.0})
                e["score"] += 1.0 / (self.rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1]["score"])[:k]
        return [{"id": doc_id, **v} for doc_id, v in top]
