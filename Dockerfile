# Single-process image: FastAPI serves the React SPA from /app/static and API routes under the same port.
# Hugging Face Docker Spaces expects a Dockerfile at the repository root; local builds use the same file:
#   docker build -t doc-ingest .
# Compose: docker/docker-compose.yml (build context is repo root).

ARG NODE_IMAGE=node:20-bookworm-slim
ARG PYTHON_IMAGE=python:3.11-slim

FROM scratch AS local-model-cache
COPY docker/model-cache/ /

FROM ${NODE_IMAGE} AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ${PYTHON_IMAGE}

ARG DEBIAN_MIRROR=http://deb.debian.org/debian
ARG DEBIAN_SECURITY_MIRROR=http://deb.debian.org/debian-security

WORKDIR /app

# Install system deps needed by python-magic and runtime health checks.
RUN sed -i \
      -e "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
      -e "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
      /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYTORCH_FIND_LINKS=
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1

COPY requirements/base.txt requirements/base.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "${PIP_INDEX_URL}" --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "${PYTORCH_FIND_LINKS}" ]; then \
      pip install --index-url "${PIP_INDEX_URL}" \
        --find-links "${PYTORCH_FIND_LINKS}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
    else \
      pip install --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "${PIP_INDEX_URL}" -r requirements/base.txt

COPY --from=frontend-builder /frontend/dist /app/static

COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY config.yaml config.yaml
COPY README.md README.md
COPY Docs/ Docs/

ARG HF_ENDPOINT=https://huggingface.co

ENV ENV=prod
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV HF_ENDPOINT=${HF_ENDPOINT}
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/hub

# Preload the exact Chinese embedding and reranking models used by the project.
# A BuildKit cache mount preserves partial downloads across interrupted builds;
# copying into /app keeps the completed models available in the final image.
RUN --mount=type=bind,from=local-model-cache,source=/,target=/local-model-cache,ro \
    --mount=type=cache,target=/root/.cache/huggingface \
    cp -a /local-model-cache/. /root/.cache/huggingface/ && \
    python scripts/cache_hf_model.py \
      --repo-id BAAI/bge-large-zh-v1.5 \
      --revision 79e7739b6ab944e86d6171e44d24c997fc1e0116 \
      --endpoint "${HF_ENDPOINT}" \
      --cache-root /root/.cache/huggingface \
      --file 1_Pooling/config.json \
      --file config.json \
      --file config_sentence_transformers.json \
      --file modules.json \
      --file pytorch_model.bin \
      --file sentence_bert_config.json \
      --file special_tokens_map.json \
      --file tokenizer.json \
      --file tokenizer_config.json \
      --file vocab.txt && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('/root/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116', local_files_only=True)" && \
    mkdir -p /app/.cache/huggingface && \
    cp -a /root/.cache/huggingface/. /app/.cache/huggingface/
RUN --mount=type=bind,from=local-model-cache,source=/,target=/local-model-cache,ro \
    --mount=type=cache,target=/root/.cache/huggingface \
    cp -a /local-model-cache/. /root/.cache/huggingface/ && \
    python scripts/cache_hf_model.py \
      --repo-id BAAI/bge-reranker-v2-m3 \
      --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
      --endpoint "${HF_ENDPOINT}" \
      --cache-root /root/.cache/huggingface \
      --file config.json \
      --file model.safetensors \
      --file sentencepiece.bpe.model \
      --file special_tokens_map.json \
      --file tokenizer.json \
      --file tokenizer_config.json && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('/root/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', local_files_only=True)" && \
    cp -a /root/.cache/huggingface/. /app/.cache/huggingface/

EXPOSE 8000

# HF Spaces runs the container as UID 1000; match that to avoid permission issues.
RUN useradd -m -u 1000 appuser && mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD sh -c 'curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1'

# PORT is honored for Hugging Face (app_port / runtime) and other platforms.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]
