# Process of building BTObot

BTObot is a local RAG-based assistant designed to answer Singapore BTO housing questions grounded in official HDB content. It was built entirely to run offline due to network restrictions and data governance concerns, using only locally available models via Ollama.

## The retrieval architecture

A parent-child retrieval (RAG) setup was chosen over naive chunking. Small chunks are used for retrieval while larger parent chunks are passed to the LLM, preserving both precision and contextual reasoning, crucial for dense HDB eligibility rules.

---

## Key Engineering Issues & Fixes

**LangChain 1.x moved the furniture.** `ParentDocumentRetriever` was removed from the main package and relocated to `langchain-classic`. The error message pointed nowhere useful. I found it by reading the migration notes and installed `langchain-classic` as an extra dependency.

---

## Performance Optimisation

Initial latency (~30-40s) was reduced significantly through:

- Keeping Ollama models resident in memory (OLLAMA_MAX_LOADED_MODELS)
- Removing redundant embedding calls in retrieval and reranking
- Reducing context window size to cut prefill cost

Finally achieve average latency of ~8-10s

## Chosen Design Tradeoffs

- Limited corpus (7 pages) to preserve retrieval quality over coverage
- UI-level citations included for transparency and user trust
- Chainlit used instead of a custom frontend for rapid development
