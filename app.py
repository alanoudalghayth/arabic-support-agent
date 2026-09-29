"""Gradio chat UI for the Arabic support agent.

Wraps the  agent in a web chat. Each browser session gets its own
conversation thread so the agent remembers context within a chat.
"""
import os
import uuid
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
from agent import agent  # noqa: E402


def respond(message, history, session_id):
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = agent.invoke({"messages": [HumanMessage(content=message)]}, config)
        return result["messages"][-1].content
    except Exception as e:
        return (f"عذراً، حدث خطأ تقني. الرجاء المحاولة مرة أخرى.\n\n"
                f"Sorry, a technical error occurred. ({type(e).__name__})")


with gr.Blocks(title="وكيل الدعم الفني | Support Agent",
               theme=gr.themes.Soft()) as demo:
    session = gr.State(lambda: str(uuid.uuid4()))

    gr.Markdown(
        """
        # وكيل خدمة العملاء | Bilingual Support Agent

        Ask in Arabic (MSA or Gulf dialect) or English. The agent searches a
        knowledge base, checks your sentiment with a fine-tuned Arabic model,
        and escalates when it can't answer.

        اسأل بالعربية أو الإنجليزية.
        """
    )

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[session],
        type="messages",
        examples=[
            ["كم يوم عندي عشان أرجع المنتج؟"],
            ["وش رسوم الشحن للرياض؟"],
            ["طلبي تأخر كثير وأنا زعلان جداً!"],
            ["What is your warranty policy?"],
        ],
        cache_examples=False,
    )

    gr.Markdown(
        "Built with LangGraph · ChromaDB · hybrid BM25+dense retrieval · "
        "a fine-tuned MARBERTv2 sentiment microservice. "
        "[Source](https://github.com/YOUR_USER/arabic-support-agent)"
    )

if __name__ == "__main__":
    demo.launch()
