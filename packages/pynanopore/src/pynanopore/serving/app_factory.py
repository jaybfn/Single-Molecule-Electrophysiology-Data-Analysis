"""FastAPI middleware and exception handlers for production services."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from pynanopore.serving.errors import error_body
from pynanopore.serving.logging_config import configure_logging
from pynanopore.serving.settings import ServiceSettings

REQUEST_ID_STATE_KEY = "request_id"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach/propagate request IDs, enforce upload size, emit access logs."""

    def __init__(self, app, settings: ServiceSettings, logger: logging.Logger):
        super().__init__(app)
        self.settings = settings
        self.logger = logger
        self.max_bytes = int(settings.max_upload_mb) * 1024 * 1024

    async def dispatch(self, request: Request, call_next: Callable):
        header = self.settings.request_id_header
        request_id = request.headers.get(header) or str(uuid.uuid4())
        request.state.request_id = request_id

        # Reject oversized bodies early when Content-Length is present
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content=error_body(
                            code="upload_too_large",
                            message=(
                                f"Request body exceeds limit of {self.settings.max_upload_mb} MB"
                            ),
                            request_id=request_id,
                            details={"max_upload_mb": self.settings.max_upload_mb},
                        ),
                        headers={header: request_id, "X-Service": self.settings.service_name},
                    )
            except ValueError:
                pass

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            self.logger.exception(
                "unhandled_error",
                extra={"service": self.settings.service_name, "request_id": request_id},
            )
            return JSONResponse(
                status_code=500,
                content=error_body(
                    code="internal_error",
                    message="Internal server error",
                    request_id=request_id,
                ),
                headers={header: request_id, "X-Service": self.settings.service_name},
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers[header] = request_id
        response.headers["X-Service"] = self.settings.service_name
        self.logger.info(
            "request_completed method=%s path=%s status=%s duration_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            extra={"service": self.settings.service_name, "request_id": request_id},
        )
        return response


def _request_id_from(request: Request) -> str | None:
    return getattr(request.state, REQUEST_ID_STATE_KEY, None)


def register_exception_handlers(app: FastAPI, settings: ServiceSettings) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = _request_id_from(request)
        # Preserve structured detail if already an envelope
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            body = detail
        else:
            code = "http_error"
            if exc.status_code == 413:
                code = "upload_too_large"
            elif exc.status_code == 502:
                code = "upstream_unreachable"
            elif exc.status_code == 400:
                code = "bad_request"
            elif exc.status_code == 404:
                code = "not_found"
            body = error_body(
                code=code,
                message=str(detail),
                request_id=request_id,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=body,
            headers={
                settings.request_id_header: request_id or "",
                "X-Service": settings.service_name,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = _request_id_from(request)
        return JSONResponse(
            status_code=422,
            content=error_body(
                code="validation_error",
                message="Request validation failed",
                request_id=request_id,
                details={"errors": exc.errors()},
            ),
            headers={
                settings.request_id_header: request_id or "",
                "X-Service": settings.service_name,
            },
        )


def enforce_upload_size(
    data: bytes, settings: ServiceSettings, request_id: str | None = None
) -> None:
    """Raise HTTP 413 if in-memory upload exceeds configured limit."""
    max_bytes = int(settings.max_upload_mb) * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=error_body(
                code="upload_too_large",
                message=f"Upload exceeds limit of {settings.max_upload_mb} MB",
                request_id=request_id,
                details={"size_bytes": len(data), "max_upload_mb": settings.max_upload_mb},
            ),
        )


def configure_service(app: FastAPI, settings: ServiceSettings) -> logging.Logger:
    """Attach logging, middleware, and exception handlers to a FastAPI app."""
    logger = configure_logging(
        service_name=settings.service_name,
        level=settings.log_level,
        json_logs=settings.log_json,
    )
    app.add_middleware(RequestContextMiddleware, settings=settings, logger=logger)
    register_exception_handlers(app, settings)
    return logger
