FROM ghcr.io/astral-sh/uv:0.12.15 AS uv

FROM python:3.12.14-slim AS runtime
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:${PATH}"
RUN apt-get update \
  && apt-get install --no-install-recommends -y libgomp1 \
  && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
RUN useradd --create-home --uid 10001 technova \
  && mkdir -p /app/artifacts /app/data/processed \
  && chown -R technova:technova /app
USER technova
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "technova_ai_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
