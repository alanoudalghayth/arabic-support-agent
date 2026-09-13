"""Turn text into vectors (embeddings) for meaning-based search."""
import os
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.needs_prefix = "e5" in model_name.lower()

    def _prep(self, texts: list[str], kind: str) -> list[str]:
        if not self.needs_prefix:
            return texts
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + t for t in texts]

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(self._prep(texts, "doc"), normalize_embeddings=True,
                                 batch_size=16, show_progress_bar=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(self._prep([text], "query"),
                                 normalize_embeddings=True)[0]
