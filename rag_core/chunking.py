"""Split documents into searchable chunks, respecting sentence boundaries."""
import re
from dataclasses import dataclass, field

SENT_END = re.compile(r"(?<=[.!?؟۔])\s+|\n{2,}")


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    meta: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_END.split(text) if s and s.strip()]


def chunk_document(text: str, doc_id: str, target_chars: int = 900,
                   overlap_chars: int = 150, meta: dict | None = None) -> list[Chunk]:
    sentences = split_sentences(text)
    raw_chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + len(s) + 1 <= target_chars:
            buf = f"{buf} {s}"
        else:
            raw_chunks.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail} {s}".strip()
    if buf:
        raw_chunks.append(buf)
    return [Chunk(text=c, doc_id=doc_id, chunk_id=f"{doc_id}::{i}", meta=meta or {})
            for i, c in enumerate(raw_chunks)]
