"""Agent tools.

Tool docstrings ARE the prompt. The model chooses tools by reading them, so
vague docstrings are the number one cause of agents that never call the right
tool. Write them as instructions to a new colleague.
"""
import os
import json
import pathlib
import datetime

import httpx
from langchain_core.tools import tool

from rag_core import VectorStore, HybridRetriever

_store = VectorStore(path="chroma_db", collection="support_kb")
_retriever = HybridRetriever(_store)

SENTIMENT_URL = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")
TICKETS = pathlib.Path("tickets.jsonl")


@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for policies, fees, timeframes, and
    procedures. Works in Arabic and English. You MUST call this before answering
    any factual question about returns, shipping, payment, warranty, or accounts.
    Returns numbered excerpts with their source ids."""
    hits = _retriever.search(query, k=4)
    if not hits:
        return "NO_RESULTS"
    return "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)


@tool
def check_customer_sentiment(text: str) -> str:
    """Analyze the emotional tone of a customer message using a fine-tuned Arabic
    sentiment model. Call this on the customer's FIRST message, and again any time
    they appear frustrated. Returns a label, a confidence score, and whether
    escalation is recommended."""
    try:
        r = httpx.post(f"{SENTIMENT_URL}/predict",
                       json={"texts": [text]}, timeout=4.0)
        r.raise_for_status()
        p = r.json()["predictions"][0]
        urgent = p["label"] == "negative" and p["confidence"] > 0.85
        return (f"sentiment={p['label']} confidence={p['confidence']:.2f} "
                f"escalate_recommended={urgent}")
    except (httpx.TimeoutException, httpx.HTTPError, KeyError, ValueError) as e:
        # Graceful degradation: an optional signal must never take down the
        # main conversational path. Return something the LLM can reason about,
        # not an exception.
        return f"SENTIMENT_UNAVAILABLE ({type(e).__name__}) — continue without it"


@tool
def escalate_to_human(summary: str, customer_language: str, urgency: str) -> str:
    """Create a ticket for a human agent. Call this when the knowledge base
    returns NO_RESULTS, when the answer is not clearly contained in the
    retrieved excerpts, when the customer is angry, or when the request involves
    refunds above policy, legal matters, or account deletion.

    Args:
        summary: 2-3 sentence summary of the issue, written in English.
        customer_language: 'ar' or 'en'.
        urgency: one of 'low', 'medium', 'high'.
    """
    ticket = {
        "id": f"ESC-{datetime.datetime.now():%Y%m%d-%H%M%S}",
        "summary": summary,
        "language": customer_language,
        "urgency": urgency,
        "created_at": datetime.datetime.now().isoformat(),
    }
    with TICKETS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    return f"Escalated successfully. Ticket id: {ticket['id']}"


TOOLS = [search_knowledge_base, check_customer_sentiment, escalate_to_human]
