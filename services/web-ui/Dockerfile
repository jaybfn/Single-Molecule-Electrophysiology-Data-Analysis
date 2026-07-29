FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GATEWAY_URL=http://gateway:8000 \
    EXAMPLE_CSV_PATH=/app/data/test.csv

WORKDIR /app

COPY services/web-ui/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY services/web-ui/app.py ./app.py
COPY data/test.csv ./data/test.csv

USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
