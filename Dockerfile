# evaluator-result-parser — HTTP server and CLI tools (see README.md)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY t4_visualizer/ ./t4_visualizer/
COPY result_parser/ ./result_parser/

# Pin t4-devkit for reproducible builds (see upstream tags on GitHub).
RUN pip install --upgrade pip && \
    pip install "git+https://github.com/tier4/t4-devkit.git@v0.6.0" && \
    pip install ".[server]" pyarrow seaborn

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data && chown appuser:appuser /data /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["t4-server", "--data-dir", "/data", "--host", "0.0.0.0", "--port", "8000"]
