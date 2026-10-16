# Agritech Answers API (step 3).
# Build from the repository root:  docker build -t agritech-api .
# Run:                             docker run -p 8000:8000 agritech-api

FROM python:3.13.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Non-root user, uid 1000 (also what Hugging Face Spaces expects).
RUN useradd --create-home --uid 1000 agritech
WORKDIR /app

# Dependencies first: this layer is reused as long as requirements do not change.
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

# Only what the API serves: code and models. No data/, no notebooks.
COPY src/ src/
COPY api/ api/
COPY models/ models/

USER agritech
EXPOSE 8000

# Models take a few seconds to load: hence the start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD \
    python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)" || exit 1

# --no-access-log: the API already logs each request with its latency (api/main.py).
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
