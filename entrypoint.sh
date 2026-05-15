#!/usr/bin/env bash
# BTObot container entrypoint
# Runs on every `docker compose up`:
#   1. Wait for Ollama to be healthy
#   2. Pull required models (no-op if already downloaded into the volume)
#   3. Build the knowledge base (skipped if ChromaDB volume is populated)
#   4. Start Chainlit
set -euo pipefail

OLLAMA="${OLLAMA_BASE_URL:-http://ollama:11434}"

# ── 1. Wait for Ollama ────────────────────────────────────────────────────────
printf "\n[BTObot] Waiting for Ollama at %s ...\n" "$OLLAMA"
until curl -sf "${OLLAMA}/api/tags" > /dev/null; do
  sleep 3
done
echo "[BTObot] Ollama is ready."

# ── 2. Pull models ────────────────────────────────────────────────────────────
# stream:false blocks until the pull completes; no-op if the model is already
# present in the ollama_models volume from a previous run.
for MODEL in qwen2.5:3b nomic-embed-text; do
  printf "[BTObot] Pulling %s (skipped if already downloaded) ...\n" "$MODEL"
  curl -sf --max-time 1800 \
    -X POST "${OLLAMA}/api/pull" \
    -H 'Content-Type: application/json' \
    -d "{\"name\":\"${MODEL}\",\"stream\":false}" > /dev/null
  printf "[BTObot] %s ready.\n" "$MODEL"
done

# ── 3. Build knowledge base ───────────────────────────────────────────────────
# Check whether ChromaDB already has data (populated in a previous run).
if [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then
  echo "[BTObot] Building knowledge base — scraping 7 HDB pages ..."
  echo "[BTObot] This takes ~5 min on first run; subsequent starts skip this step."
  python ingest.py
else
  echo "[BTObot] Knowledge base already exists — skipping ingest."
fi

# ── 4. Start Chainlit ─────────────────────────────────────────────────────────
echo "[BTObot] Starting on http://localhost:8000"
exec chainlit run app.py --host 0.0.0.0 --port 8000
