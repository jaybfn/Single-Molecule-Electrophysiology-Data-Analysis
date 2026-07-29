FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY packages/pynanopore ./packages/pynanopore
COPY services/psd-service/app.py ./services/psd-service/app.py

RUN pip install --upgrade pip \
    && pip install ".[services,viz]" \
    && useradd --create-home --uid 10001 appuser

USER appuser
WORKDIR /app/services/psd-service
EXPOSE 8003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
