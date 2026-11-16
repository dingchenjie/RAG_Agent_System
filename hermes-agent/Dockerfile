FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# 默认不装 torch/语义模型；需要 bge 时：--build-arg INSTALL_EXTRAS=semantic
ARG INSTALL_EXTRAS=""

WORKDIR /app

COPY pyproject.toml README.md ./
COPY hermes ./hermes
RUN pip install --upgrade pip \
 && if [ -n "$INSTALL_EXTRAS" ]; then pip install ".[$INSTALL_EXTRAS]"; else pip install .; fi

COPY knowledge_base ./knowledge_base
COPY eval ./eval
COPY scripts ./scripts

ENV HERMES_DB_PATH=/data/hermes.db \
    HERMES_LLM_PROVIDER=mock \
    HERMES_UPLOAD_DIR=/data/uploads
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "hermes.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
