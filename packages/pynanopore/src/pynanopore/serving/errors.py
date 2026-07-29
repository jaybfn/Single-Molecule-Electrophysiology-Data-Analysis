"""Standard error response envelopes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail = Field(...)


def error_body(
    *,
    code: str,
    message: str,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            details=details,
        )
    ).model_dump()
