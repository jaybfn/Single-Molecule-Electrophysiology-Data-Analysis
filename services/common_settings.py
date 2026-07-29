"""Shared settings and response helpers for microservices."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"
    max_upload_mb: int = 100
    request_id_header: str = "X-Request-ID"


class GatewaySettings(ServiceSettings):
    event_service_url: str = "http://event-service:8001"
    stats_service_url: str = "http://stats-service:8002"
    psd_service_url: str = "http://psd-service:8003"
    http_timeout_s: float = 120.0
