"""Retrieval evaluation. Produces the ablation table for the README."""
import json
import pathlib
from dotenv import load_dotenv
from rag_core import VectorStore, HybridRetriever

load_dotenv()
GOLDEN = pathlib.Path("evals/golden_ar.jsonl")


def evaluate(retriever, rows, k=5, **kwargs):
    hits, rr = 0, 0.0
    by_lang = {}
    for row in rows:
        got = [h["id"] for h in retriever.search(row["q"], k=k, **kwargs)]
        relevant = set(row["relevant_ids"])
        rank = next((i + 1 for i, g in enumerate(got) if g in relevant), None)
        hit = rank is not None
        hits += hit
        rr += 1.0 / rank if rank else 0.0
        b = by_lang.setdefault(row.get("lang", "?"), [0, 0])
        b[0] += hit
        b[1] += 1
    n = len(rows)
    return {"recall": hits / n, "mrr": rr / n,
            "by_lang": {k_: v[0] / v[1] for k_, v in sorted(by_lang.items())}}


def main():
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    store = VectorStore(path="chroma_db", collection="support_kb")
    retriever = HybridRetriever(store)

    configs = [("Dense only", {"dense_only": True}),
               ("BM25 only", {"bm25_only": True}),
               ("Hybrid (RRF)", {})]

    langs = sorted({r.get("lang", "?") for r in rows})
    print(f"\nEvaluating on {len(rows)} questions, k=5\n")
    header = f"{'Config':<16}{'Recall':<9}{'MRR':<8}" + "".join(f"{l:<13}" for l in langs)
    print(header)
    print("-" * len(header))
    for name, kwargs in configs:
        m = evaluate(retriever, rows, k=5, **kwargs)
        line = f"{name:<16}{m['recall']:<9.3f}{m['mrr']:<8.3f}"
        line += "".join(f"{m['by_lang'].get(l, 0):<13.3f}" for l in langs)
        print(line)
    print()


if __name__ == "__main__":
    main()
