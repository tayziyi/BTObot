# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# curl is used in entrypoint.sh to health-check Ollama and pull models.
# playwright install --with-deps handles all Chromium OS-level libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so this layer is cached on code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright Chromium ───────────────────────────────────────────────────────
# --with-deps installs OS packages needed by Chromium in the same step.
RUN playwright install --with-deps chromium

# ── Application source ────────────────────────────────────────────────────────
COPY . .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
