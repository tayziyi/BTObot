"""
BTObot — Data Ingestion Pipeline
=================================
Reads URLs from urls.txt, scrapes each page, applies parent-child chunking,
embeds the child chunks with mxbai-embed-large via Ollama, and persists
everything to ChromaDB + a LocalFileStore for parent documents.

Usage:
    python ingest.py                  # ingest urls.txt
    python ingest.py --urls my.txt    # custom URL file
    python ingest.py --reset          # wipe existing DB first, then ingest
"""

import argparse
import asyncio
import base64
import hashlib
import os
import shelve
import shutil
import time
from pathlib import Path
from uuid import uuid4

import bs4
import requests as _http
from playwright.async_api import async_playwright

# Must be set before any langchain import to suppress the USER_AGENT warning
os.environ.setdefault("USER_AGENT", "BTObot/1.0 (+https://github.com/BTObot)")

from stores import LocalFileStore                               # parent doc store
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    OLLAMA_BASE_URL, EMBED_MODEL,
    CHROMA_PATH, COLLECTION_NAME, PARENT_STORE_PATH,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    SCRAPE_DELAY_SECONDS, VISION_MODEL, VISION_MIN_PX,
    VISION_MAX_IMAGES_PER_PAGE, VISION_CACHE_PATH,
)

# ---------------------------------------------------------------------------
# Vision helpers  (image-to-text via Ollama multimodal model)
# ---------------------------------------------------------------------------

def _vision_cache_key(identifier: str) -> str:
    """Stable shelve key: model name + SHA-256 of the identifier string."""
    return hashlib.sha256(f"{VISION_MODEL}:{identifier}".encode()).hexdigest()


def _call_vision_model(image_bytes: bytes, cache_key: str) -> str:
    """
    Look up *cache_key* in the vision cache first; if not found, send
    the screenshot to the Ollama vision model and cache the result.
    Returns "" on any error.
    """
    # --- cache read ---
    try:
        with shelve.open(VISION_CACHE_PATH) as db:
            if cache_key in db:
                cached = db[cache_key]
                print(f"    [vision] cache hit ({len(cached)} chars)")
                return cached
    except Exception:
        pass

    # --- model call ---
    b64 = base64.b64encode(image_bytes).decode()
    try:
        resp = _http.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": (
                    "Extract all text visible in this image exactly as written. "
                    "If it is a table, reproduce each row. "
                    "If it is a chart or diagram, list all labels and values. "
                    "Output only the extracted content — no explanations."
                ),
                "images": [b64],
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except Exception as exc:
        print(f"    [vision] model call failed: {exc}")
        return ""

    # --- cache write ---
    if text:
        try:
            with shelve.open(VISION_CACHE_PATH) as db:
                db[cache_key] = text
        except Exception:
            pass

    return text


async def _extract_image_text(page) -> str:
    """
    Screenshot every large <img> and <svg> element on the page, send each
    to the vision model, and return the concatenated extracted text.

    Skips:
      - Images smaller than VISION_MIN_PX on either axis (icons / bullets)
      - Duplicate <img src> URLs (same asset referenced multiple times)
      - Images beyond VISION_MAX_IMAGES_PER_PAGE (stops early to save time)
    Cache:
      Results are persisted by image URL or content hash so that
      re-ingests avoid re-calling the model for unchanged images.
    """
    if not VISION_MODEL:
        return ""

    captions: list[str] = []
    seen_srcs: set[str] = set()
    count = 0
    limit = VISION_MAX_IMAGES_PER_PAGE  # 0 means unlimited

    # --- <img> elements -------------------------------------------------------
    imgs = await page.query_selector_all("img")
    for img in imgs:
        if limit and count >= limit:
            break
        src = (await img.get_attribute("src")) or ""
        if src in seen_srcs:
            continue
        bbox = await img.bounding_box()
        if not bbox or bbox["width"] < VISION_MIN_PX or bbox["height"] < VISION_MIN_PX:
            continue
        seen_srcs.add(src)
        try:
            shot = await img.screenshot()
            key = _vision_cache_key(src or hashlib.md5(shot).hexdigest())
            text = _call_vision_model(shot, key)
            if text:
                print(f"    [vision] <img> ({int(bbox['width'])}×{int(bbox['height'])}px) → {len(text)} chars")
                captions.append(text)
                count += 1
        except Exception as exc:
            print(f"    [vision] screenshot failed for img: {exc}")

    # --- <svg> elements -------------------------------------------------------
    svgs = await page.query_selector_all("svg")
    for svg in svgs:
        if limit and count >= limit:
            break
        bbox = await svg.bounding_box()
        if not bbox or bbox["width"] < VISION_MIN_PX or bbox["height"] < VISION_MIN_PX:
            continue
        try:
            shot = await svg.screenshot()
            key = _vision_cache_key(hashlib.md5(shot).hexdigest())
            text = _call_vision_model(shot, key)
            if text:
                print(f"    [vision] <svg> ({int(bbox['width'])}×{int(bbox['height'])}px) → {len(text)} chars")
                captions.append(text)
                count += 1
        except Exception as exc:
            print(f"    [vision] screenshot failed for svg: {exc}")

    return "\n\n".join(captions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Tags to keep when extracting text (everything else is stripped)
KEEP_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "li", "td", "th", "dt", "dd"}


def load_urls(filepath: str) -> list[str]:
    """Read non-empty, non-comment lines from a URL file."""
    lines = Path(filepath).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


async def _fetch_page(playwright, url: str) -> str:
    """
    Open a single URL in a headless Chromium browser.
    - Waits for 'load' (DOM + resources ready), then an extra 4s for React.
    - Uses a realistic desktop viewport and UA to avoid bot-detection.
    - Falls back gracefully on timeout.
    """
    browser = await playwright.chromium.launch(headless=True)
    try:
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-SG",
        )
        page = await context.new_page()
        await page.goto(url, wait_until="load", timeout=45_000)
        # Give React/Angular time to populate the DOM after initial load
        await page.wait_for_timeout(4_000)

        html = await page.content()

        # Extract text from images/SVGs via vision model (if configured)
        if VISION_MODEL:
            print(f"    [vision] scanning images on {url} …")
            image_text = await _extract_image_text(page)
            if image_text:
                # Inject extracted image text as <p> tags so BeautifulSoup
                # picks them up through the normal KEEP_TAGS extraction path.
                injection = "\n".join(
                    f"<p>{line}</p>"
                    for line in image_text.splitlines()
                    if line.strip()
                )
                html = html.replace("</body>", f"{injection}\n</body>", 1)

        return html
    finally:
        await browser.close()


def _html_to_text(html: str, url: str) -> str:
    """Extract meaningful text from rendered HTML using BeautifulSoup."""
    soup = bs4.BeautifulSoup(html, "lxml")

    # Remove noisy elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    # Collect text from informative tags
    parts: list[str] = []
    for tag in soup.find_all(KEEP_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)

    return " ".join(parts)


def scrape_urls(urls: list[str]) -> list[Document]:
    """
    Scrape each URL with headless Chromium (Playwright) and wait for
    network-idle so JavaScript-rendered content (HDB.gov.sg SPA) is
    fully loaded before extraction.
    """
    docs: list[Document] = []

    async def _scrape_all() -> None:
        async with async_playwright() as pw:
            for i, url in enumerate(urls):
                if i > 0:
                    await asyncio.sleep(SCRAPE_DELAY_SECONDS)
                try:
                    html = await _fetch_page(pw, url)
                    text = _html_to_text(html, url)
                    text = " ".join(text.split())   # normalise whitespace

                    if len(text) < 100:
                        print(f"  ⚠  {url}  — too little content ({len(text)} chars), skipping")
                        print(f"     Preview: {text[:120]!r}")
                        continue

                    # Try to extract the page title
                    soup = bs4.BeautifulSoup(html, "lxml")
                    title = soup.title.string.strip() if soup.title else url

                    docs.append(Document(
                        page_content=text,
                        metadata={"source": url, "title": title},
                    ))
                    print(f"  ✓  {url}  ({len(text):,} chars)")
                except Exception as exc:
                    print(f"  ✗  {url}  — {exc}")

    asyncio.run(_scrape_all())
    return docs


def reset_stores() -> None:
    """Delete Chroma DB and parent store directories."""
    for path in (CHROMA_PATH, PARENT_STORE_PATH):
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
            print(f"  Removed: {path}")


def build_retriever(embeddings: OllamaEmbeddings) -> ParentDocumentRetriever:
    """Instantiate the LangChain ParentDocumentRetriever backed by Chroma + LocalFileStore."""
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    store = LocalFileStore(PARENT_STORE_PATH)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )


# ---------------------------------------------------------------------------
# Batched ingestion helper (shows live progress, saves incrementally)
# ---------------------------------------------------------------------------

def _ingest_batched(
    retriever: ParentDocumentRetriever,
    docs: list[Document],
    batch_size: int = 20,
) -> int:
    """
    Replicates ParentDocumentRetriever.add_documents() but:
      • Splits parents from children explicitly so we can show counts.
      • Embeds + writes children in batches (batch_size chunks at a time).
        This means partial work is saved to Chroma even if the process dies.
      • Prints a live progress line after every batch.
    """
    id_key: str = retriever.id_key  # default: "doc_id"

    # ── 1. Split raw pages into parent chunks ──────────────────────────────
    parent_splitter = retriever.parent_splitter
    parent_docs: list[Document] = (
        parent_splitter.split_documents(docs) if parent_splitter else list(docs)
    )
    print(f"  Parent chunks : {len(parent_docs)}")

    # ── 2. Tag each parent with a unique ID and persist to the file store ──
    parent_ids = [str(uuid4()) for _ in parent_docs]
    for doc, pid in zip(parent_docs, parent_ids):
        doc.metadata[id_key] = pid
    retriever.docstore.mset(list(zip(parent_ids, parent_docs)))
    print(f"  Saved {len(parent_docs)} parent docs to disk")

    # ── 3. Split parents into child chunks, inheriting the parent ID ───────
    child_splitter = retriever.child_splitter
    child_docs: list[Document] = []
    for pdoc in parent_docs:
        children = child_splitter.split_documents([pdoc])
        for child in children:
            child.metadata[id_key] = pdoc.metadata[id_key]
        child_docs.extend(children)
    total = len(child_docs)
    print(f"  Child chunks  : {total}  (batch size: {batch_size})")
    print()

    # ── 4. Embed + store in batches ────────────────────────────────────────
    t0 = time.time()
    for i in range(0, total, batch_size):
        batch = child_docs[i : i + batch_size]
        retriever.vectorstore.add_documents(batch)  # embeds + persists to Chroma
        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"  [{done:>4}/{total}]  {done/total*100:5.1f}%  "
            f"elapsed {elapsed:5.0f}s  ETA {eta:5.0f}s",
            flush=True,
        )

    print(f"\n  All {total} child chunks embedded and saved.")
    return total


# ---------------------------------------------------------------------------
# Main ingestion entry-point
# ---------------------------------------------------------------------------

def ingest(urls_file: str = "urls.txt", reset: bool = False) -> None:
    if reset:
        print("\n[1/5] Resetting existing stores …")
        reset_stores()
    else:
        print("\n[1/5] Skipping reset (use --reset to wipe existing data)")

    print("\n[2/5] Loading URLs …")
    urls = load_urls(urls_file)
    if not urls:
        raise SystemExit(f"No URLs found in '{urls_file}'. Add some and retry.")
    print(f"  Found {len(urls)} URL(s)")

    print("\n[3/5] Scraping pages …")
    docs = scrape_urls(urls)
    if not docs:
        raise SystemExit("No documents loaded. Check network access and URLs.")
    print(f"\n  Scraped {len(docs)} document(s)")

    print(f"\n[4/5] Initialising embeddings (Ollama / {EMBED_MODEL}) …")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # Warm-up: single embed call to trigger model load before timing the real work
    print("  Warming up embedding model (first load may take 10-20s) …")
    embeddings.embed_query("hello")
    print("  Model ready.")

    print("\n[5/5] Chunking → embedding → storing …")
    retriever = build_retriever(embeddings)
    child_count = _ingest_batched(retriever, docs, batch_size=20)

    # Summary
    vstore_count = retriever.vectorstore._collection.count()
    print(
        f"\n✅  Ingestion complete!"
        f"\n   Child chunks (Chroma):  {vstore_count}"
        f"\n   Chroma DB path:         {CHROMA_PATH}"
        f"\n   Parent store path:      {PARENT_STORE_PATH}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTObot ingestion pipeline")
    parser.add_argument(
        "--urls", default="urls.txt", help="Path to URL list (default: urls.txt)"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Wipe existing DB before ingesting"
    )
    args = parser.parse_args()
    ingest(urls_file=args.urls, reset=args.reset)
