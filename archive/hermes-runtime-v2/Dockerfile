FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install fastapi uvicorn pydantic httpx sse-starlette mcp

COPY hermes ./hermes
COPY knowledge_base ./knowledge_base
COPY eval ./eval
COPY scripts ./scripts

ENV HERMES_DB_PATH=/data/hermes.db HERMES_LLM_PROVIDER=mock
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "hermes.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
