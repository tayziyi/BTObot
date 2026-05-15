"""
BTObot — Central configuration.
All tuneable knobs live here; override any value via a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Suppress LangChain USER_AGENT warning for web scraping
os.environ.setdefault("USER_AGENT", "BTObot/1.0 (+https://github.com/BTObot)")

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama registry has qwen2.5:3b (no 4b tag exists); override via LLM_MODEL in .env
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "mxbai-embed-large")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME: str = "bto_children"      # stores embedded child chunks

# ---------------------------------------------------------------------------
# Parent document store  (persisted to disk via LocalFileStore)
# ---------------------------------------------------------------------------
PARENT_STORE_PATH: str = os.getenv("PARENT_STORE_PATH", "./parent_store")

# ---------------------------------------------------------------------------
# Chunking  — parent chunks are larger (context window for the LLM),
#             child chunks are smaller (precision for embedding search).
# ---------------------------------------------------------------------------
PARENT_CHUNK_SIZE: int    = int(os.getenv("PARENT_CHUNK_SIZE",    "1000"))
PARENT_CHUNK_OVERLAP: int = int(os.getenv("PARENT_CHUNK_OVERLAP", "100"))
CHILD_CHUNK_SIZE: int     = int(os.getenv("CHILD_CHUNK_SIZE",     "300"))
CHILD_CHUNK_OVERLAP: int  = int(os.getenv("CHILD_CHUNK_OVERLAP",  "50"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# Retrieve this many children from Chroma, expand to their parents, then rerank.
TOP_K_CHILDREN: int = int(os.getenv("TOP_K_CHILDREN", "10"))
# Final number of parent docs returned to the LLM.
TOP_N_FINAL: int    = int(os.getenv("TOP_N_FINAL", "3"))

# Max characters from each parent chunk injected into the LLM prompt.
# Keeps the prompt short → faster pre-fill on CPU.  Full text still shown
# in the Chainlit side panel (retrieved from the original doc object).
CONTEXT_CHARS_PER_SOURCE: int = int(os.getenv("CONTEXT_CHARS_PER_SOURCE", "700"))

# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
SCRAPE_DELAY_SECONDS: float = float(os.getenv("SCRAPE_DELAY_SECONDS", "1.5"))

# Vision model for extracting text from images during ingest.
# Leave empty ("") to skip image extraction (default — faster ingest).
# Set to an Ollama multimodal model to enable:
#   ollama pull llava:7b       (~4.7 GB, best accuracy for infographics)
#   ollama pull minicpm-v      (~5.5 GB, strong document OCR)
# Then set:  VISION_MODEL=llava:7b  in your .env
VISION_MODEL: str = os.getenv("VISION_MODEL", "")

# Minimum image dimension (px) to send to the vision model.
# Images smaller than this on either axis are treated as icons and skipped.
VISION_MIN_PX: int = int(os.getenv("VISION_MIN_PX", "150"))

# Maximum images to process per page (0 = unlimited, not recommended on CPU).
# The most informative infographics tend to appear first, so 5 is usually enough.
VISION_MAX_IMAGES_PER_PAGE: int = int(os.getenv("VISION_MAX_IMAGES_PER_PAGE", "5"))

# Shelve path for vision result cache.
# Extracted text is stored by image URL / content hash so that re-ingests
# (python ingest.py --reset) skip images that were already processed.
VISION_CACHE_PATH: str = os.getenv("VISION_CACHE_PATH", "./vision_cache")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are BTObot, a knowledgeable and friendly AI assistant \
specialising in Singapore's Build-To-Order (BTO) housing scheme for young couples.

Use ONLY the numbered context passages below to answer the question. \
Cite each passage you draw from using [1], [2], or [3] inline. \
If the answer is not present in the context, say: \
"I don't have specific information on that — please check the HDB website directly \
(https://www.hdb.gov.sg)."

Guidelines:
- Be concise, warm, and practical.
- Highlight key eligibility criteria, deadlines, and dollar amounts when relevant.
- Never fabricate figures or policy details not present in the context."""
