"""
BTObot — Chainlit Frontend
===========================
Run with:
    chainlit run app.py

Features
--------
- Streaming LLM responses via ChatOllama
- Source citations shown as collapsible side-panels after each answer
- Multi-turn conversation history (last 3 exchanges kept for LLM context)
- Step indicator while retrieving so the user sees progress
- Graceful error messages if the knowledge base has not been ingested yet
"""

from __future__ import annotations

import asyncio
import time

import chainlit as cl
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import (
    OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
    SYSTEM_PROMPT, TOP_N_FINAL, CONTEXT_CHARS_PER_SOURCE,
)
from retriever import BTORetriever

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, reused across all sessions.
# ---------------------------------------------------------------------------
_retriever: BTORetriever | None = None
_llm: ChatOllama | None = None


def get_retriever() -> BTORetriever:
    global _retriever
    if _retriever is None:
        _retriever = BTORetriever()
    return _retriever


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            streaming=True,
        )
    return _llm


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------

async def _tick_timer(step: cl.Step, label: str) -> None:
    """Updates a Step's name with elapsed seconds every second until cancelled.
    Gives the user a live 'Thought for Xs…' counter during LLM generation.
    """
    t0 = time.time()
    while True:
        await asyncio.sleep(1)
        elapsed = int(time.time() - t0)
        step.name = f"{label} {elapsed}s…"
        await step.update()


async def _warmup_models() -> None:
    """Pre-load both Ollama models into RAM (fire-and-forget at chat start).
    Without this, the FIRST user query pays a ~26s model-load penalty per model.
    """
    try:
        r = get_retriever()
        loop = asyncio.get_event_loop()
        # Load embedding model
        await loop.run_in_executor(
            None, lambda: r._embeddings.embed_query("warmup")
        )
        # Load LLM (1-token generation forces Ollama to load the weights)
        warmup_llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_predict=1,
        )
        await warmup_llm.ainvoke([HumanMessage(content="hi")])
        get_llm()   # initialise the singleton
    except Exception:
        pass        # non-fatal — first query will just be slower


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start() -> None:
    """Show welcome message immediately, then warm both Ollama models in background."""
    cl.user_session.set("history", [])

    try:
        r = get_retriever()
        kb_status = f"Knowledge base ready — {r.child_count:,} indexed chunks."
    except Exception as exc:
        kb_status = (
            f"⚠️  Knowledge base not ready: {exc}\n"
            "Run `python ingest.py` first, then restart the app."
        )

    # Send welcome immediately — don't block on model loading
    await cl.Message(
        content=(
            "## 👋 Welcome to BTObot!\n\n"
            "I'm your AI guide to Singapore's **Build-To-Order (BTO)** housing scheme, "
            "especially for young couples looking to apply for their first flat.\n\n"
            "I can help with:\n"
            "- 🏠 **Eligibility** — income ceiling, citizenship, family nucleus\n"
            "- 💰 **Grants** — Enhanced Housing Grant (EHG), CPF Housing Grant, PHG\n"
            "- 📋 **Application process** — balloting, selection, signing\n"
            "- 📍 **Flat types & locations** — 2-room Flexi, 3-room, 4-room, 5-room\n"
            "- ⏱️ **Timelines** — waiting times, key collection\n\n"
            f"_{kb_status}_\n\n"
            "**What would you like to know?**"
        )
    ).send()

    # Warm both models in the background so the first real query is fast
    asyncio.create_task(_warmup_models())


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Process a user query through the full RAG pipeline and stream the answer."""
    query = message.content.strip()
    if not query:
        return

    retriever = get_retriever()

    # ------------------------------------------------------------------
    # Step 1: Retrieve
    # ------------------------------------------------------------------
    async with cl.Step(name="🔍 Searching knowledge base…") as step:
        try:
            docs = retriever.retrieve(query)
        except Exception as exc:
            await cl.Message(
                content=(
                    f"⚠️ Retrieval failed: `{exc}`\n\n"
                    "Please ensure:\n"
                    "1. `python ingest.py` has been run at least once.\n"
                    "2. Ollama is running (`ollama serve`).\n"
                    "3. The `nomic-embed-text` model is pulled (`ollama pull nomic-embed-text`)."
                )
            ).send()
            return

        step.output = (
            f"Found {len(docs)} relevant passage(s)."
            if docs
            else "No relevant passages found; answering from general knowledge."
        )

    # ------------------------------------------------------------------
    # Step 2: Build prompt  (context truncated to reduce LLM pre-fill time)
    # ------------------------------------------------------------------
    history: list[dict] = cl.user_session.get("history", [])

    # Truncate each source to CONTEXT_CHARS_PER_SOURCE for the prompt.
    # Full text is still shown in the side panels below.
    context_blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "HDB website")
        title  = doc.metadata.get("title",  source)
        snippet = doc.page_content[:CONTEXT_CHARS_PER_SOURCE]
        context_blocks.append(f"[{i}] {title}\nURL: {source}\n\n{snippet}")
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    # Build message list: system + last 1 exchange of history + current question.
    # Keeping history short (2 messages) avoids long pre-fill on CPU.
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history[-2:]:          # last 1 exchange = 2 messages
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=content))

    # Inject retrieved context just before the current question
    user_turn = (
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    messages.append(HumanMessage(content=user_turn))

    # ------------------------------------------------------------------
    # Step 3: Stream LLM response with live thinking timer
    # ------------------------------------------------------------------
    llm = get_llm()
    response_text = ""
    t_start = time.time()

    msg = cl.Message(content="")
    async with cl.Step(name="💭 Thinking 0s…") as gen_step:
        timer = asyncio.create_task(_tick_timer(gen_step, "💭 Thinking"))
        try:
            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    await msg.stream_token(token)
                    response_text += token
        finally:
            timer.cancel()
            elapsed = int(time.time() - t_start)
            gen_step.name = f"💭 Thought for {elapsed}s"

    # ------------------------------------------------------------------
    # Step 4: Attach source panels
    # ------------------------------------------------------------------
    if docs:
        elements: list[cl.Text] = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "")
            title  = doc.metadata.get("title",  f"Source {i}")
            short_title = title[:60] + ("…" if len(title) > 60 else "")
            elements.append(
                cl.Text(
                    name=f"[{i}] {short_title}",
                    content=(
                        f"**Source:** [{source}]({source})\n\n"
                        f"{doc.page_content}"
                    ),
                    display="side",
                )
            )
        msg.elements = elements

    await msg.send()

    # ------------------------------------------------------------------
    # Step 5: Update conversation history
    # ------------------------------------------------------------------
    history.append({"role": "user",      "content": query})
    history.append({"role": "assistant", "content": response_text})
    cl.user_session.set("history", history[-2:])   # keep last 1 exchange
