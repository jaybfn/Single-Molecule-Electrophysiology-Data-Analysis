"""Service configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """Base settings shared by all HTTP services."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "service"
    log_level: str = "INFO"
    log_json: bool = True
    request_id_header: str = "X-Request-ID"
    max_upload_mb: int = 100
    http_timeout_s: float = 120.0


class GatewaySettings(ServiceSettings):
    service_name: str = "gateway"
    event_service_url: str = "http://event-service:8001"
    stats_service_url: str = "http://stats-service:8002"
    psd_service_url: str = "http://psd-service:8003"
    downstream_timeout_s: float = 120.0
