# BTObot

A locally-hosted AI assistant for Singapore's Build-To-Order (BTO) housing scheme. Designed for young couples navigating eligibility, grants, financial planning, and the application process — all sourced directly from HDB's official website, with no data leaving your machine.

---

## What It Does

BTObot answers natural-language questions about BTO housing by:

1. **Retrieving** the most relevant passages from a pre-indexed snapshot of HDB's official guidance pages
2. **Generating** a grounded, cited answer using a local LLM
3. **Showing** the exact source passages in collapsible side panels so you can verify every claim

Example questions it handles well:
- *"Am I eligible for the Enhanced Housing Grant as a couple?"*
- *"What is the income ceiling for a 4-room BTO flat?"*
- *"How does the balloting process work?"*
- *"What CPF and cash is needed at key collection?"*

---

## Why

BTO eligibility rules, grant amounts, and application timelines change frequently. Searching HDB's website manually is slow and the answers are spread across many pages. BTObot consolidates seven official HDB pages into a single conversational interface that:

- Runs **100% locally** — no query or document leaves your machine
- **Cites sources inline** ([1][2][3]) so answers are verifiable
- Refuses to fabricate — if context is absent, it says so and points to HDB directly
- Works inside a corporate network without requiring external AI API access

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  INGEST  (offline, run once)                                     │
│                                                                  │
│  urls.txt                                                        │
│      │                                                           │
│      ▼  Playwright (headless Chromium)                           │
│  Raw HTML  ──BeautifulSoup──▶  Plain text  (~7 pages)            │
│      │                                                           │
│      ▼  RecursiveCharacterTextSplitter  (1200 chars / 120 ovlp)  │
│  Parent chunks  ──────────────────────▶  parent_store/  (shelve) │
│      │                                                           │
│      ▼  RecursiveCharacterTextSplitter  (600 chars / 80 ovlp)    │
│  Child chunks  ──nomic-embed-text──▶  768-dim vectors            │
│      │                                                           │
│      ▼                                                           │
│  ChromaDB  (chroma_db/)   ~369 child vectors persisted to disk   │
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
│      ▼  qwen2.5:3b via ChatOllama  (num_predict=400)             │
│  Streamed answer  +  source side panels                          │
└──────────────────────────────────────────────────────────────────┘
```

### Key design choices

| Component | Choice | Reason |
|---|---|---|
| LLM | `qwen2.5:3b` via Ollama | Runs on CPU; 32k context; strong instruction following |
| Embeddings | `nomic-embed-text` via Ollama | 274 MB, 768-dim, faster than mxbai-embed-large on CPU |
| Vector store | ChromaDB (persistent) | Zero-config, embedded SQLite, survives restarts |
| Parent-child chunking | Child=600 chars for search, Parent=1200 chars for LLM | Precise retrieval without losing surrounding context |
| Reranker | Cosine similarity (numpy) |  |
| Scraper | Playwright headless Chromium | HDB.gov.sg is a JS-rendered React SPA; `requests` returns empty |
| Frontend | Chainlit | Native async streaming, built-in step indicators, side panels |
| Context truncation | 700 chars per source in prompt | Reduces LLM pre-fill time from ~110s to ~33s on Intel CPU |

---

## Results

Tested against the 7 ingested HDB pages (knowledge base snapshot: May 2026):

| Query type | Quality |
|---|---|
| Direct eligibility questions (income ceiling, citizenship) | ✅ Accurate, cited |
| Grant amounts (EHG, CPF Housing Grant, PHG) | ✅ Accurate if in KB |
| Application process steps | ✅ Good summary |
| Complex multi-condition eligibility ("if my income is X and Y…") | ⚠️ Occasional reasoning errors |
| BTO launch dates / specific projects | ❌ Not in KB — correctly declines |
| Resale flat prices | ❌ Not in KB — correctly declines |

**Latency** (Intel Core i7, macOS, CPU-only):

| Stage | Time |
|---|---|
| Model warmup at first connect | ~50s (background) |
| embed_query (warm model) | ~0.2s |
| ChromaDB search | ~0.03s |
| LLM generation (≤400 tokens) | ~50–80s |
| **Total per query (warm)** | **~50–80s** |

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.12.x | 3.13+ breaks onnxruntime; 3.12 required |
| Ollama | ≥ 0.23.3 | [ollama.com](https://ollama.com) |
| `qwen2.5:3b` | — | `ollama pull qwen2.5:3b` |
| `nomic-embed-text` | — | `ollama pull nomic-embed-text` |
| Playwright Chromium | — | `playwright install chromium` |

Python packages: see `requirements.txt`. Notable additions not in requirements.txt that may be needed on corporate networks:
- `pip-system-certs` — injects macOS/Windows system CA certificates into Python's SSL bundle (required if behind an SSL-inspecting proxy such as Zscaler)

---

## Setup & Running

There are two ways to run BTObot: **Docker** (recommended — one command, no manual setup) or **local** (faster iteration if you are actively developing).

---

### Option A — Docker (recommended)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
git clone <repo>
cd BTObot
cp .env.example .env          # edit if needed (defaults work out of the box)
docker compose up --build
```

Open **http://localhost:8000** once you see:
```
[BTObot] Starting on http://localhost:8000
```

**First run** (~10–15 min): pulls `qwen2.5:3b` + `nomic-embed-text`, scrapes and embeds the HDB pages.  
**Subsequent runs** (`docker compose up`): volumes already populated, ready in ~30 seconds.

#### GPU acceleration

Docker's bundled Ollama container runs CPU-only on all platforms. Use the override files below to enable GPU:

| Platform | GPU available? | Command |
|---|---|---|
| Any OS | ❌ CPU only (default) | `docker compose up --build` |
| Apple Silicon Mac | ✅ Metal via native Ollama | See below |
| Windows / Linux with NVIDIA | ✅ CUDA passthrough | See below |

**Apple Silicon Mac** — run Ollama natively (Metal GPU), btobot in Docker:
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

### Option B — Local (manual)

Use this if you are actively developing and want faster iteration without rebuilding the image.

#### 1. Create virtual environment

```bash
git clone <repo>
cd BTObot
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

#### 2. Configure environment

```bash
cp .env.example .env
# Edit .env to override defaults (model names, chunk sizes, etc.)
```

#### 3. Pull Ollama models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

#### 4. Start Ollama

```bash
# OLLAMA_MAX_LOADED_MODELS=2 keeps both models in RAM simultaneously,
# eliminating the ~26s model-swap overhead between embedding and generation.
OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_FLASH_ATTENTION=1 ollama serve
```

#### 5. Ingest knowledge base (first time only)

```bash
python ingest.py
# Scrapes 7 HDB pages, embeds ~275 child chunks.
# Takes 15–25 min on Intel CPU. Progress is saved every 20 chunks.

# To wipe and re-ingest from scratch:
python ingest.py --reset
```

#### 6. Launch BTObot

```bash
chainlit run app.py
# Opens at http://localhost:8000
```

The welcome message appears immediately. Both Ollama models warm up in the background (~50s on Intel CPU). The first query takes ~50–80s; subsequent queries are faster once models are resident.

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

---

## Configuration Reference

All values can be overridden in `.env`:

| Variable | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:3b` | Ollama LLM model tag |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model tag |
| `LLM_TEMPERATURE` | `0.3` | Lower = more factual, less creative |
| `PARENT_CHUNK_SIZE` | `1200` | Characters per parent chunk stored in shelve |
| `CHILD_CHUNK_SIZE` | `600` | Characters per child chunk indexed in ChromaDB |
| `TOP_K_CHILDREN` | `10` | Child chunks fetched from ChromaDB per query |
| `TOP_N_FINAL` | `3` | Parent chunks passed to LLM after reranking |
| `CONTEXT_CHARS_PER_SOURCE` | `700` | Characters of each source injected into LLM prompt |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `PARENT_STORE_PATH` | `./parent_store` | Shelve file path for parent documents |

**To speed up generation further**, swap to a smaller model (no re-ingest needed):
```
LLM_MODEL=qwen2.5:1.5b   # ~2× faster generation, slightly lower quality
```

**Changing `EMBED_MODEL` requires re-ingesting** (`python ingest.py --reset`) because vector dimensions change.

---

## Evaluation

### Chosen methodology: human eval on a golden test set + LLM-as-judge for citation faithfulness

**The question that actually matters** for BTObot is not "does the response sound good?" but "does it correctly tell someone whether they qualify for a grant?" That is a factual, high-stakes question with a right answer derivable from HDB policy. That narrows the field considerably.

**Why not a benchmark?** No labelled dataset of BTO eligibility questions exists. Constructing one from scratch still requires humans to write questions and verify ground-truth answers — that is human eval, just with extra steps.

**Why not LLM-as-judge for factual accuracy?** Two problems. First, a judge model cannot verify a claim against HDB policy it has never seen — it can only assess whether the claim sounds plausible, which is exactly the failure mode we are trying to catch. Second, LLMs are well-documented to reward fluent, confident answers regardless of correctness, and eligibility rules are full of specific figures ($14,000 income ceiling, $80,000 grant cap) where a confident wrong number is worse than no answer.

**Why not pure ablation?** Ablation is useful for understanding which components contribute what — parent-child vs flat chunking, reranking vs no reranking — but it answers a relative question ("does component X help?") rather than an absolute one ("is the system trustworthy enough to use?"). Ablations would be a useful second layer after establishing a baseline, not a substitute for it.

**What is actually proposed:**

1. **Golden test set (~30 questions), hand-curated across four categories:**
   - Eligibility lookups with clear yes/no ground truth (*"A couple earning $10,500/month — do they qualify for the EHG?"*)
   - Specific figure retrieval (*"What is the income ceiling for a 5-room flat under the Family Grant?"*)
   - Process/timeline questions (*"When does the HLE letter need to be obtained by?"*)
   - Out-of-scope questions that should trigger a deflection (*"What is the resale levy for a 5-room flat?"*)

2. **Human scoring** on the first three categories: pass/fail per question, judged by someone who has read the source HDB pages. The scoring criterion is factual correctness, not fluency. Target: ≥80% pass rate; deflection rate of 100% on out-of-scope questions.

3. **LLM-as-judge scoped to citation faithfulness** on the first two categories: given the source passage and the answer, does the cited passage actually entail the claim attached to it? This is textual entailment — a task LLMs handle reliably — and it does not require domain knowledge. It can be run automatically across all 30 questions and gives a signal on hallucination that human eval alone would miss in subtle cases.

**What the numbers would tell you.** A high pass rate with low citation faithfulness means the model is getting the right answers by luck or general knowledge, not grounding — a deployment risk. A low pass rate with high citation faithfulness means the retrieval is surfacing the right passages but the LLM is misreading them — a different problem. The combination distinguishes between retrieval failures, generation failures, and grounding failures in a way that a single aggregate score cannot.

### Sample test set (15 questions)

Questions are labelled by category: **E** = eligibility lookup, **F** = figure retrieval, **P** = process/timeline, **O** = out-of-scope (expect deflection).

| # | Category | Question | What a correct answer requires |
|---|---|---|---|
| 1 | E | A couple with a combined monthly income of $10,500 — do they qualify for the Enhanced Housing Grant? | Correctly apply the EHG income ceiling ($9,000 for families; $4,500 for singles) and return a no |
| 2 | E | Can a couple where one party is a Singapore PR and the other is a Singapore Citizen apply for a BTO flat? | Confirm yes under the SC/PR household scheme, with the SC as the primary applicant |
| 3 | E | Is a 34-year-old single person eligible to apply for a BTO flat? | Confirm yes under the Singles scheme (≥35 years old) — return a no with the correct age threshold |
| 4 | E | If I previously received a housing subsidy as a second-timer, can I still apply for another BTO flat? | Identify second-timer status and correctly state the additional conditions and resale levy obligations |
| 5 | E | A couple where one party previously owned a private property — are they eligible for the Family Grant? | Apply the eligibility rule requiring both applicants to not have previously owned subsidised housing or private property |
| 6 | F | What is the income ceiling for a first-timer family applying for a 4-room BTO flat? | Return the correct figure ($14,000/month for most estates; $7,000 for PLH flats) |
| 7 | F | What is the maximum Enhanced Housing Grant amount a first-timer family can receive? | Return $120,000 (for households earning ≤$1,500/month) |
| 8 | F | How much is the Proximity Housing Grant for a family living with parents in the same town? | Return $30,000 (living with) vs $20,000 (living near) distinction |
| 9 | F | What is the minimum cash downpayment required when taking an HDB housing loan? | Return that HDB loan requires no minimum cash — up to 80% LTV, payable fully from CPF |
| 10 | F | What CPF Ordinary Account savings can be used for — purchase price, stamp duty, or both? | Confirm CPF OA can be used for the purchase price, legal fees, and stamp duty |
| 11 | P | What is the correct sequence of steps from BTO application to key collection? | Return the steps in order: apply → ballot → book flat → sign agreement → pay → key collection |
| 12 | P | When does the HDB Loan Eligibility (HLE) letter need to be obtained relative to the flat selection appointment? | State it must be obtained before the flat selection appointment |
| 13 | P | What happens if I miss my flat selection appointment slot? | State the consequence (forfeit queue position / treated as having declined) |
| 14 | O | What is the resale levy amount for selling a 5-room subsidised flat? | Deflect — resale levy specifics are not in the knowledge base |
| 15 | O | How is the Additional Buyer's Stamp Duty calculated for a second residential property? | Deflect — stamp duty is not covered in the knowledge base |

---

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

### Performance
- **CPU-only on Intel Mac**: Ollama does not support GPU (Metal) on Intel Macs — only on Apple Silicon (M1/M2/M3). All inference runs on CPU.
- **~50–80s per query**: At 5–8 tokens/second on Intel Core i7, a 400-token answer takes 50–80 seconds. This is a hardware constraint that cannot be resolved in software beyond model switching.
- **Model warmup**: On the first query after a cold start (or after Ollama unloads models due to the 5-minute keep-alive timeout), each model takes ~26s to load from disk, adding up to ~52s to latency.
- **Ingest takes 15–25 min**: All 275 child chunks are embedded via Ollama's HTTP API sequentially on CPU. Progress is saved after every 20-chunk batch, so a crash can be recovered by re-running `python ingest.py` (without `--reset`).

### Session behaviour
- **No persistence across browser tabs**: Chainlit sessions are tab-scoped. Conversation history does not carry over to a new tab or a page refresh.
- **History limited to 1 exchange**: Only the last question–answer pair is retained in the LLM context to keep prompt length short on CPU. This means the bot has no memory of earlier turns in the same conversation.
- **Single-user**: There is no authentication. Anyone with network access to `localhost:8000` can use the bot.

---

## Deployment Considerations

**Who and where.** The natural operator is HDB itself, or a government digital services team, hosting BTObot on an internal server or a GPU-enabled cloud instance behind VPN — accessible to officers handling public enquiries, or as a public-facing self-service tool on MyHDBPage. For a proof-of-concept at team scale, a single Apple Silicon Mac mini (M2 Pro, ~$1,500) runs both the LLM and embedding model at roughly 50–80 tok/s with no recurring cost. At 500 queries/day (a modest public deployment), an AWS `g4dn.xlarge` (NVIDIA T4, ~$0.53/hr on-demand) runs comfortably under $50/month and handles ~20 concurrent users.

**What to monitor.** Track the "I don't have information" deflection rate — a spike signals that HDB has restructured pages or added new policy that isn't in the KB. P95 response latency and citation click-through rate (do users actually verify sources?) are the other two metrics worth a dashboard.

**The risk that keeps you up at night.** HDB revises grant amounts, income ceilings, and eligibility rules at every budget cycle — sometimes mid-year. Unlike a search engine, BTObot gives no visual signal that its knowledge base is six months old. A user could receive a confidently-cited answer with an income ceiling that is now $1,500 higher, miss a grant they qualify for, or submit an application based on superseded criteria. Given that grants reach $120,000 or more, the cost of stale data is not a UX problem — it's a financial harm. A mandatory scheduled re-ingest (e.g., weekly cron) and a visible "knowledge last updated" timestamp on every response are non-negotiable before any public-facing release.
