# Agritech Answers interface (step 4).
# Build from the repository root:  docker build -f app/Dockerfile -t agritech-app .
# Usually started with the API by:  docker compose up

FROM python:3.13.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    API_URL=http://localhost:8000

RUN useradd --create-home --uid 1000 agritech
WORKDIR /app

COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# The interface only: no model, no API code.
COPY app/ app/
COPY .streamlit/ .streamlit/

USER agritech
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD \
    python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)" || exit 1

CMD ["sh", "-c", "streamlit run app/app.py --server.port ${PORT} --server.address 0.0.0.0"]
