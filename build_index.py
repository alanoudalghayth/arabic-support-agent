"""Build the Chroma index from data/kb/. Run locally; commit the output."""
import pathlib
import re
import shutil
from dotenv import load_dotenv
from rag_core import chunk_document, VectorStore

load_dotenv()
KB_DIR = pathlib.Path("data/kb")
DB_PATH = "chroma_db"


def parse_frontmatter(text: str):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, m.group(2)


def main():
    if pathlib.Path(DB_PATH).exists():
        shutil.rmtree(DB_PATH)
    store = VectorStore(path=DB_PATH, collection="support_kb")
    total = 0
    for path in sorted(KB_DIR.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("doc_id", path.stem)
        chunks = chunk_document(body, doc_id=doc_id,
                                meta={"title": meta.get("title", doc_id)})
        store.add(chunks)
        total += len(chunks)
        print(f"  {path.name:24s} -> {len(chunks)} chunks")
    print(f"\nIndexed {total} chunks from {len(list(KB_DIR.glob('*.md')))} documents")
    print(f"Collection count: {store.count()}")


if __name__ == "__main__":
    main()
