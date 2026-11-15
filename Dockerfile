# catalog-mcp: Streamable HTTP server image (CPU-only torch, embedding model baked in).
#
#   docker build -t catalog-mcp .
#   docker run --rm -v catalog-data:/data catalog-mcp build    # one-time; slow on CPU (see README)
#   docker run --rm -p 8000:8000 -v catalog-data:/data -e CATALOG_MCP_TOKEN=... catalog-mcp

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/opt/hf \
    CATALOG_MCP_HOME=/data

WORKDIR /app
# CPU wheels for torch first, so the CUDA build (several GB) is never pulled in.
RUN uv pip install --system "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install --system --no-sources .

RUN useradd --create-home --uid 1000 app && mkdir -p /data /opt/hf && chown app:app /data /opt/hf
USER app
# Bake in the DuckDB full-text-search extension and the embedding model: no downloads at serve time.
RUN python -c "import duckdb; duckdb.connect().execute('INSTALL fts')" \
 && python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5')"

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

ENTRYPOINT ["catalog-mcp"]
CMD ["serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
