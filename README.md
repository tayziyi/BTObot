# BTObot

AI assistant for Singapore's Build-To-Order (BTO) housing scheme. Designed for young couples navigating eligibility, grants, financial planning, and the application process, all sourced directly from HDB's official website as well as trusted blogs curated.

---

## What BTObot Does

It answers natural-language questions about BTO housing by:

1. **Retrieving** the most relevant context from a pre-indexed snapshot of HDB's official guidance pages and latest trustworthy blogs
2. **Generating** a grounded, cited answer using a local LLM
3. **Showing** the exact source context in collapsible side panels so you can verify every claim

Example questions:
- *"Who are eligible for the Enhanced Housing Grant?"*
- *"What is the income ceiling for a 4-room BTO flat?"*
- *"Show me the process of buying a BTO flat."*

---

## Why BTObot

The current chatbot available in HDB portal does not provide a direct answer to users but shares relevant links. Additionally, BTO eligibility rules, grant amounts, and application timelines change frequently. Searching HDB's website manually is slow and the answers are spread across many pages. BTObot consolidates the latest official HDB pages into a single conversational interface that:

- Answers user enquires directly with retrieved context
- Refuses to fabricate — if context is absent, it says so and points to HDB directly
- **Cites sources inline** ([1][2][3]) so answers are verifiable


---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  INGEST  (offline, run once)                                     │
│                                                                  │
│  urls.txt                                                        │
│      │                                                           │
│      ▼  Playwright (headless Chromium)                           │
│  Raw HTML  ──BeautifulSoup──▶  Plain text                        │
│      │                                                           │
│      ▼  RecursiveCharacterTextSplitter  (1200 chars / 120 ovlp)  │
│  Parent chunks  ──────────────────────▶  parent_store/  (shelve) │
│      │                                                           │
│      ▼  RecursiveCharacterTextSplitter  (600 chars / 80 ovlp)    │
│  Child chunks  ──nomic-embed-text──▶  768-dim vectors            │
│      │                                                           │
│      ▼                                                           │
│  ChromaDB  (chroma_db/)                                          │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  QUERY  (per user message)                                       │
│                                                                  │
│  User question                                                   │
│      │                                                           │
│      ▼  nomic-embed-text  (Ollama, ~0.2s when warm)              │
│  768-dim query vector                                            │
│      │                                                           │
│      ▼  ChromaDB cosine similarity  (k=10 child chunks)          │
│  Top-10 child chunks                                             │
│      │                                                           │
│      ▼  Parent expansion  (look up parent IDs in shelve)         │
│  Unique parent chunks  (typically 3–5)                           │
│      │                                                           │
│      ▼  Cosine-similarity reranker  (numpy, no download needed)  │
│  Top-3 parent chunks  (first 700 chars each → LLM prompt)        │
│      │                                                           │
│      ▼  qwen2.5:3b via ChatOllama                                │
│  Streamed answer  +  source side panels                          │
└──────────────────────────────────────────────────────────────────┘
```

## Key design choices

| Component | Choice | Reason |
|---|---|---|
| LLM | `qwen2.5:3b` via Ollama | 32k context; strong instruction following; lightweight model|
| Embeddings | `nomic-embed-text` via Ollama | 274 MB, 768-dim, lightweight to run locally |
| Vector store | ChromaDB (persistent) | Zero-config, embedded SQLite, survives restarts |
| Parent-child chunking | Child=600 chars for search, Parent=1200 chars for LLM | Precise retrieval without losing surrounding context |
| Reranker | Cosine similarity (numpy) |  |
| Scraper | Playwright headless Chromium | HDB.gov.sg is a JS-rendered React SPA; `requests` returns empty |
| Frontend | Chainlit | Native async streaming, built-in step indicators, side panels |

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.12.x | 3.13+ breaks onnxruntime; 3.12 required |
| Ollama | ≥ 0.23.3 | [ollama.com](https://ollama.com) |
| `qwen2.5:3b` | — | `ollama pull qwen2.5:3b` |
| `nomic-embed-text` | — | `ollama pull nomic-embed-text` |
| Playwright Chromium | — | `playwright install chromium` |

Python packages: see `requirements.txt`.

---

## Setup & Running

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

BTObot can be ran with **Docker** in one command and no manual setup.

Docker's bundled Ollama container runs CPU-only on all platforms. Use the override files below to enable GPU:

---
**Apple Silicon Mac** — run Ollama natively (Metal GPU), BTObot in Docker:
```bash
# Terminal 1 — keep running
OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_FLASH_ATTENTION=1 ollama serve

# Terminal 2
docker compose -f docker-compose.yml -f docker-compose.mac-gpu.yml up --build
```

**Windows / Linux with NVIDIA** — requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):
```bash
docker compose -f docker-compose.yml -f docker-compose.nvidia-gpu.yml up --build
```

#### Day-to-day commands (after first build)

| Situation | Command |
|---|---|
| Normal start (CPU) | `docker compose up` |
| Normal start (Apple Silicon GPU) | `ollama serve` + `docker compose -f docker-compose.yml -f docker-compose.mac-gpu.yml up` |
| Normal start (NVIDIA GPU) | `docker compose -f docker-compose.yml -f docker-compose.nvidia-gpu.yml up` |
| After editing source code | add `--build` to whichever command above |
| Stop (keep data) | `docker compose down` |
| Wipe DB and re-ingest | `docker compose down -v` then `docker compose up` |


---

## Project Structure

```
BTObot/
├── app.py                        # Chainlit frontend — streaming, citations, history
├── retriever.py                  # Retrieval pipeline: embed → search → expand → rerank
├── ingest.py                     # Scrape → chunk → embed → store (batched, with progress)
├── stores.py                     # Custom shelve-based LocalFileStore for parent documents
├── config.py                     # All tuneable parameters (reads from .env)
├── urls.txt                      # URL seed list for ingestion
├── requirements.txt
├── .env / .env.example
├── .gitignore
├── Dockerfile                    # Python 3.12 + Playwright Chromium
├── docker-compose.yml            # Default: bundled Ollama (CPU) + btobot
├── docker-compose.mac-gpu.yml    # Override: native Ollama (Metal GPU) on Apple Silicon
├── docker-compose.nvidia-gpu.yml # Override: NVIDIA GPU passthrough on Windows/Linux
├── entrypoint.sh                 # Container startup: wait → pull models → ingest → serve
├── chainlit.md                   # Chainlit welcome page config
├── chroma_db/                    # ChromaDB vector store (created by ingest.py, git-ignored)
└── parent_store/                 # Shelve store for parent chunks (created by ingest.py, git-ignored)
```

## Evaluation

### Chosen Methodology: Human Eval on a Golden Test Set 

BTObot is a RAG system for public-sector policy guidance. We are evaluating - factual correctness, citation faithfulness.

**Golden Test Set**

1. 15 Questions hand-curated across four categories:
   - Eligibility lookups with clear yes/no ground truth (E) 
   - Specific figure retrieval (F)
   - Process/timeline questions (P)
   - Out-of-scope questions that should trigger a deflection (O) 

2. **Human scoring** on the first three categories: pass/fail per question, judged by someone who has read the source HDB pages. The scoring criterion is factual correctness, not fluency. Target: ≥80% pass rate; deflection rate of 100% on out-of-scope questions.

| #  | Category | Question                                                                                  | Model Response Summary                                                          | Result    |
| -- | -------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------- |
| 1  | E        | Does a couple with combined monthly income of $10,500 qualify for Enhanced Housing Grant? | Refused and redirected to HDB website instead of applying income threshold rule | ❌ Fail    |
| 2  | E        | Can a couple where one party is SC and the other PR apply for a BTO flat?                 | Correctly stated eligibility under SC/PR scheme                                 | ✅ Pass    |
| 3  | E        | Can two single Singapore Citizens (≥35) jointly apply for a BTO flat?                     | Correctly stated eligibility under Joint Singles Scheme                         | ✅ Pass    |
| 4  | E        | Can two Singapore PRs apply together for a BTO flat?                                      | Incorrectly refused to answer (missing eligibility rule SC requirement)         | ❌ Fail    |
| 5  | F        | What is the monthly household income ceiling for purchasing a new flat?                   | Correctly retrieved income ceiling figure                                       | ✅ Pass    |
| 6  | F        | How much grant can I receive with combined monthly income of $10,000?                     | Incorrect / incomplete grant computation                                        | ❌ Fail    |
| 7  | F        | What income thresholds determine Enhanced CPF Housing Grant amount?                       | Correct explanation of tiered thresholds                                        | ✅ Pass    |
| 8  | F        | How is household income calculated?                                                       | Correct explanation of income computation method                                | ✅ Pass    |
| 9  | P        | What is the correct sequence of steps from BTO application to key collection?             | Correct full process flow                                                       | ✅ Pass    |
| 10 | P        | What is the process of buying a BTO?                                                      | Correct high-level process explanation                                          | ✅ Pass    |
| 11 | P        | When can I apply for BTO balloting?                                                       | Hallucinated incorrect timing / eligibility condition                           | ❌ Fail    |
| 12 | P        | When does the HLE letter need to be obtained relative to flat selection?                  | Incorrect refusal (did not apply known rule)                                    | ❌ Fail    |
| 13 | O        | What is the resale levy amount for selling a 5-room subsidised flat?                      | Correctly deflected (out-of-scope)                                              | ✅ Deflect |
| 14 | O        | How is ABSD calculated for a second residential property?                                 | Correctly deflected (out-of-scope)                                              | ✅ Deflect |
| 15 | O        | Which BTO project is the best investment?                                                 | Correctly deflected (out-of-scope)                            | ✅ Deflect |

Overall Evaluation
- Total Pass Rate (E + F + P): 7 / 12 → 58.3%
- Out-of-Scope Deflection: 100% (Target Achieved)
- Main failure modes:
   1. Over-refusal (deflection instead of answering grounded eligibility questions)
   2. Hallucinated or incomplete process reasoning 

## Known Limitations

### Knowledge coverage
- **Static snapshot**: The knowledge base is a one-time scrape of 7 HDB pages. New BTO launches, policy updates, grant amount changes, and new eligibility rules are **not reflected** until you re-run `python ingest.py --reset`.
- **7 pages only**: Resale levy, ethnic integration policy, premium flat details, CPF housing limits, stamp duty, and many other topics are **not in the KB**. The LLM correctly declines and redirects to HDB.gov.sg, but it cannot look things up live.
- **No table/structure preservation**: HDB pages contain eligibility tables with income ranges and grant amounts. BeautifulSoup extracts these as prose. Row/column relationships (e.g., income bracket X → grant amount Y) may be garbled in the text.

### Retrieval quality
- **Bi-encoder reranking**: The reranker uses the same `nomic-embed-text` model that performed the initial vector search. This adds a second scoring pass but cannot capture deep query–document interaction the way a cross-encoder (e.g., ms-marco-MiniLM) would. Queries with indirect or multi-hop phrasing may surface weaker passages.
- **Context truncation**: Only the first 700 characters of each parent chunk (out of 1200) are sent to the LLM, to reduce pre-fill time. Eligibility conditions that appear later in a chunk may be missed.
- **No query expansion or HyDE**: The query is embedded as-is. Misspellings, abbreviations, or phrasing that differs from HDB's wording can reduce retrieval precision.

### LLM reasoning
- **3B parameter model**: `qwen2.5:3b` is capable for extraction and summarisation from provided context but can make errors on complex multi-condition eligibility questions (e.g., "if my income is $6,000 and I am a PR married to a SC, what grants apply?"). Always verify against the cited source.
- **Hallucination risk**: Although instructed to cite only provided context, the model may occasionally blend in general knowledge. The inline citations [1][2][3] allow you to check whether each claim appears in the source.
- **Context window budget**: With `num_predict=400`, responses are capped at ~400 tokens. Long eligibility explanations may be cut off.

---

## Deployment Considerations

**Who and where.** The natural operator is HDB itself, or a government digital services team, hosting BTObot on an internal server or a GPU-enabled cloud instance behind VPN — accessible to officers handling public enquiries, or as a public-facing self-service tool on MyHDBPage. For a proof-of-concept at team scale, a single Apple Silicon Mac mini (M2 Pro, ~$1,500) runs both the LLM and embedding model at roughly 50–80 tok/s with no recurring cost. At 500 queries/day (a modest public deployment), an AWS `g4dn.xlarge` (NVIDIA T4, ~$0.53/hr on-demand) runs comfortably under $50/month and handles ~20 concurrent users.

**What to monitor.** Track the "I don't have information" deflection rate — a spike signals that HDB has restructured pages or added new policy that isn't in the KB. P95 response latency and citation click-through rate (do users actually verify sources?) are the other two metrics worth a dashboard.

**The risk that keeps you up at night.** HDB revises grant amounts, income ceilings, and eligibility rules at every budget cycle — sometimes mid-year. Unlike a search engine, BTObot gives no visual signal that its knowledge base is six months old. A user could receive a confidently-cited answer with an income ceiling that is now $1,500 higher, miss a grant they qualify for, or submit an application based on superseded criteria. Given that grants reach $120,000 or more, the cost of stale data is not a UX problem — it's a financial harm. A mandatory scheduled re-ingest (e.g., weekly cron) and a visible "knowledge last updated" timestamp on every response are non-negotiable before any public-facing release.
