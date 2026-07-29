"""Shared settings (compat shim → pynanopore.serving)."""

from pynanopore.serving.settings import GatewaySettings, ServiceSettings

__all__ = ["ServiceSettings", "GatewaySettings"]
