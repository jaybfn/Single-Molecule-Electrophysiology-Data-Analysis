FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY packages/pynanopore ./packages/pynanopore
COPY services/event-service/app.py ./services/event-service/app.py

RUN pip install --upgrade pip \
    && pip install ".[services,viz]" \
    && useradd --create-home --uid 10001 appuser

USER appuser
WORKDIR /app/services/event-service
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
