"""Structured logging setup."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter for container log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": getattr(record, "service", None),
            "request_id": getattr(record, "request_id", None),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Drop nulls for cleaner lines
        payload = {k: v for k, v in payload.items() if v is not None}
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    *, service_name: str, level: str = "INFO", json_logs: bool = True
) -> logging.Logger:
    """Configure root-ish service logger; returns the service logger."""
    logger = logging.getLogger(service_name)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Quiet noisy libs a bit
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    return logger
