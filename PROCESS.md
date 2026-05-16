# Process of building BTObot

BTObot is a local RAG-based assistant designed to answer Singapore BTO housing questions grounded in official HDB content. It was built entirely to run locally using only models served through Ollama — no external API calls, no data leaving the machine.

The first real decision was whether to use a hosted LLM API (OpenAI, Claude, Gemini) or run models locally. The choice was local. Sending housing-related queries — which may contain personal eligibility details — to a third-party model introduces a data-handling concern: query contents pass through an external service. Running locally sidesteps this: no API key, no external dependency, no data leaving the machine.

That decision shaped every downstream dependency: only models available through Ollama's registry, only Python libraries that don't need to phone home, and no runtime model downloads from HuggingFace.

### Choosing the LLM

On CPU, model size directly translates to generation speed. Several models were evaluated:

| Model | Size | Speed (CPU) | Instruction following |
|---|---|---|---|
| `llama3.2:3b` | 2.0 GB | ~6 tok/s | Good |
| `qwen2.5:1.5b` | 1.0 GB | ~12 tok/s | Weak on complex conditions |
| `qwen2.5:3b` | 1.9 GB | ~5–8 tok/s | Strong |
| `phi3.5-mini` | 2.2 GB | ~5 tok/s | Good but verbose |

`qwen2.5:3b` was chosen. BTO eligibility answers require the model to read a passage, identify the relevant condition, apply it to the user's situation, and not fabricate. The 1.5b variant consistently introduced general knowledge not present in the provided context. `qwen2.5:3b` held the grounding constraint more reliably, and its 32k context window gave room for prompt engineering without overflow concerns.

### Choosing the embedding model: mxbai-embed-large → nomic-embed-text

The initial embedding model was `mxbai-embed-large`. It produces 1024-dimensional vectors and scores well on MTEB retrieval benchmarks. The problem was speed: on a warm CPU, a single `embed_query` call took ~0.4–0.6 seconds. In the retrieval pipeline this added up — and ingest (embedding ~300+ child chunks sequentially) took noticeably longer than expected.

Switching to `nomic-embed-text` (274 MB, 768-dim) brought `embed_query` down to ~0.2 seconds warm — roughly 2–3× faster. The vector dimension drop from 1024 to 768 required wiping and re-ingesting the ChromaDB collection (vector dimensions must be consistent), but retrieval quality on BTO content was comparable. HDB's text is domain-specific and does not need the extra headroom that mxbai's higher dimensionality provides.

The switch required updating `EMBED_MODEL` in `.env` and running `python ingest.py --reset`. It also required confirming the ChromaDB collection was re-initialised at the new dimension — leaving the old 1024-dim collection in place causes a silent mismatch at query time.

---

## The retrieval architecture

A parent-child retrieval (RAG) setup was chosen over naive chunking. Small chunks are used for retrieval while larger parent chunks are passed to the LLM, preserving both precision and contextual reasoning, crucial for dense HDB eligibility rules.

**Why naive RAG fails here.** A naive setup — chunk each page into equal-size pieces, embed them, retrieve top-k — was the obvious starting point. The problem with HDB content is that eligibility conditions are dense and conditional:

> "Your household income must not exceed $14,000/month, unless you are applying for a Prime Location Public Housing flat, in which case the ceiling is $7,000/month..."

A short chunk (300 chars) is precise for embedding — it surfaces the right paragraph for a query about income ceilings. But 300 chars is not enough context for the LLM to reason about; surrounding conditions are cut off. A long chunk (1200 chars) gives the LLM enough context but makes embedding less precise — the vector now represents a paragraph mixing multiple eligibility topics, diluting the similarity signal.

Parent-child retrieval resolves this: embed small **child chunks** (600 chars) for precise search, retrieve and pass the corresponding large **parent chunk** (1200 chars) to the LLM. Embedding precision from the child; reasoning context from the parent. LangChain's `ParentDocumentRetriever` implements this pattern directly.

**Chunk sizes were tuned iteratively:**
- Started at 1000 (parent) / 300 (child) — too many chunks, slow ingest
- Settled on 1200 / 600 — ~50% fewer chunks, faster ingest, same retrieval quality
- Overlaps set to 120 / 80 to avoid losing context at chunk boundaries

### Web scraping: requests → Playwright

The first attempt used Python `requests` + BeautifulSoup directly against HDB URLs. The result was an almost-empty HTML skeleton — no eligibility text, no grant amounts. HDB's website is a React SPA: the server returns a bare HTML shell and content is injected by JavaScript after page load. `requests` cannot execute JavaScript.

The fix was Playwright — a headless Chromium browser that renders the full page and returns the populated DOM.

- `wait_until="networkidle"` (waits until no network activity for 500ms) was tried first but caused timeouts on pages with analytics pings that fire continuously.
- `wait_until="load"` + a fixed `page.wait_for_timeout(4000)` (4 seconds) was more reliable. React components sometimes finish rendering after `DOMContentLoaded`, so the buffer was necessary.
- A realistic desktop User-Agent string was set — HDB's CDN appeared to serve reduced content to non-browser UAs in testing.

---

## Key Engineering Issues & Fixes

**LangChain 1.x moved the furniture.** `ParentDocumentRetriever` was removed from the main package and relocated to `langchain-classic`. The error message pointed nowhere useful. I found it by reading the migration notes and installed `langchain-classic` as an extra dependency.

**The reranker: flashrank → numpy cosine similarity.** The original plan was to use flashrank (a lightweight wrapper around `ms-marco-MiniLM-L-12-v2`) as a cross-encoder reranker. Cross-encoders score query–document pairs jointly and produce better ranking than cosine similarity alone. Flashrank downloads its ONNX model from HuggingFace on first use.

On the first `import flashrank`, the process hung for 30 seconds then threw a connection error — `huggingface.co` was not reachable from the network. Workarounds attempted:
- Pre-downloading the ONNX weights and copying them locally: flashrank's cache path is not easily overridable
- Self-hosted mirror: none available
- `sentence-transformers` directly: also downloads from HuggingFace at runtime

The replacement was cosine similarity using the already-loaded `nomic-embed-text` vectors — no new model download, no new dependency, just numpy arithmetic on vectors already in memory. It provides marginal reranking benefit (second scoring pass with query vector vs parent chunk vector rather than child chunk vector) but cannot capture query–document interaction the way a cross-encoder would.

**SSL certificate issues.** HTTPS calls from Python were failing with `SSL: CERTIFICATE_VERIFY_FAILED`. The machine was behind a proxy performing SSL inspection — intercepting TLS connections and re-signing certificates with its own CA. Python's `requests` and `urllib` use a bundled OpenSSL certificate store (via certifi) that does not include the proxy CA. Fix: `pip install pip-system-certs`, which patches Python's SSL context to use the macOS system keychain. One install, no configuration. Listed in requirements but is a no-op on machines not behind an SSL-inspecting proxy.

---

## Performance Optimisation

Initial latency was reduced significantly through two targeted fixes.

**Fix 1 — Keeping Ollama models resident in memory (`OLLAMA_MAX_LOADED_MODELS`).** Ollama's default keep-alive is 5 minutes: after inactivity it unloads models from RAM. Setting `OLLAMA_MAX_LOADED_MODELS=2` keeps both models loaded simultaneously. Combined with a background warmup call triggered when the user opens the chat (while the welcome message is visible), the models are almost always hot by the time the first query arrives.

**Fix 2 — Removing redundant embedding calls in retrieval and reranking.** The query vector was being computed twice: once inside `ParentDocumentRetriever.invoke()` (to search ChromaDB) and again inside the reranker (for cosine scoring). Fix: embed the query once, pass the vector to both stages. This required refactoring the retriever to call `vectorstore.similarity_search_by_vector()` instead of the string-based `similarity_search()`.

The LLM generation time averaging 10s is an irreducible hardware constraint. The optimisations eliminated everything else within reach.

---

## Chosen Design Tradeoffs

- **Limited corpus (7 pages) to preserve retrieval quality over coverage.** When unrelated BTO topics compete in the embedding space, similarity scores compress and top-k results become noisier. Seven closely-scoped pages — eligibility, grants, loans, financial planning, and the application process — gave clean retrieval. Pages were expanded to 18 as confidence in the pipeline grew.

- **UI-level citations included for transparency and user trust.** The `[1][2][3]` inline citations tell the user which source was used. Full-text side panels let the user read the actual passage and verify the claim. For a financial decision involving grants of up to $120,000, "trust but verify" is the right design stance. Citations were non-trivial to implement in Chainlit but worth the effort.

- **700 chars in the LLM prompt, 1200 chars in the UI panels.** Truncating the LLM context was a deliberate quality-vs-speed tradeoff. The full parent chunk is always available in the UI so the user never loses the source. Sending all 1200 chars to the LLM doubles pre-fill time for marginal improvement — the crux of an eligibility condition almost always appears in the first 700 characters.

- **Chainlit used instead of a custom frontend for rapid development.** Options considered: Gradio (no native collapsible panels, streaming workarounds needed), Streamlit (synchronous model doesn't pair well with async LLM streaming), custom React + FastAPI (full control but a week of work minimum). Chainlit won because it is natively async, has built-in `cl.Step` for thinking indicators, built-in `cl.Text` for collapsible citation panels, and one-line streaming. The "💭 Thinking Xs…" live timer was added purely for UX — on a 60-second query a static spinner feels broken.

- **ChromaDB over alternatives.** Pinecone (cloud-hosted), Weaviate (separate server process), Qdrant (more setup than needed), FAISS (in-memory only, no built-in persistence). ChromaDB requires zero configuration, creates an embedded SQLite-backed store in a local directory, and persists automatically. For ~400 vectors on a single node, performance is not a differentiating factor.

- **Everything configurable through `.env`.** All tunable parameters — chunk sizes, model names, retrieval k values, context truncation — live in `config.py` and are overridable through `.env`. This made experimentation fast: swapping the embedding model or tuning chunk sizes was a one-line change, not a code change.
