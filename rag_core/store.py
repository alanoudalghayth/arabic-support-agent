"""ChromaDB vector store: holds document embeddings and searches them."""
import chromadb
from .embeddings import Embedder
from .chunking import Chunk


class VectorStore:
    def __init__(self, path: str = "chroma_db", collection: str = "kb",
                 embedder: Embedder | None = None):
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder or Embedder()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed_docs([c.text for c in chunks])
        self.col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            metadatas=[{"doc_id": c.doc_id, **c.meta} for c in chunks],
        )

    def query(self, text: str, k: int = 5) -> list[dict]:
        qv = self.embedder.embed_query(text)
        r = self.col.query(query_embeddings=[qv.tolist()], n_results=k)
        return [{"id": i, "text": d, "meta": m, "score": 1 - dist}
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0],
                                         r["metadatas"][0], r["distances"][0])]

    def all_docs(self) -> list[tuple[str, str]]:
        r = self.col.get()
        return list(zip(r["ids"], r["documents"]))

    def count(self) -> int:
        return self.col.count()
