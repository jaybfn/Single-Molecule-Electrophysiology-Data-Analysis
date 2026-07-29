# PR: Production hardening (logging, limits, readiness)

**Branch:** `feature/production-hardening` (or current working branch) → `main`  
**Version:** 2.4.0

## Summary

Hardens the FastAPI microservices for production-like operation:

1. Structured JSON logging + **`X-Request-ID`** propagation (gateway → downstream)
2. Upload size / timeout limits via env-backed settings + clearer error envelopes
3. Docker Compose readiness (`service_healthy`, `start_period`) and shared ops env vars
4. Dependency lockfile for reproducible installs

## Motivation

After science features landed, the highest leverage work is operational reliability: correlatable logs, bounded uploads, predictable timeouts, and compose that waits for healthy backends before starting the gateway/UI.

## What’s changed

### Shared serving package (`pynanopore.serving`)
- `ServiceSettings` / `GatewaySettings` (env: `LOG_LEVEL`, `LOG_JSON`, `MAX_UPLOAD_MB`, `HTTP_TIMEOUT_S`, `DOWNSTREAM_TIMEOUT_S`, …)
- `RequestContextMiddleware`: request ID attach/echo, early `413` on oversized `Content-Length`, access logs
- Standard error envelope: `{ "error": { "code", "message", "request_id", "details?" } }`
- `configure_service(app, settings)` wired into gateway, event, stats, PSD services

### Gateway
- Forwards `X-Request-ID` on all downstream `httpx` calls
- Enforces upload size before proxying ABF/CSV uploads
- Uses `downstream_timeout_s` for analysis calls

### Compose
- `depends_on: condition: service_healthy` for gateway → analysis services and web-ui → gateway
- Healthchecks gain `start_period`
- Ops env vars passed through to services

### Lockfile
- `requirements.lock` generated from `pyproject.toml` extras for reproducible CI/Docker builds
- CI and service Dockerfiles install with `-c requirements.lock`

### Version
- Package bumped to **2.4.0**

## How to test

```bash
pip install -e ".[dev,viz,services]"
pytest tests/test_serving.py tests/test_services.py

# Optional: exercise limits
# MAX_UPLOAD_MB=1 uvicorn … then POST a >1MB file → 413 envelope

docker compose up --build
# Confirm gateway waits until event/stats/psd are healthy
curl -i http://localhost:8000/health
# Echoed X-Request-ID:
curl -i -H "X-Request-ID: demo-1" http://localhost:8001/health
```

## Checklist

- [x] Request ID middleware + JSON logs
- [x] Error envelopes for HTTP / validation / upload size
- [x] Upload + timeout settings on services + gateway
- [x] Compose healthy depends_on + start_period
- [x] Unit tests for serving helpers
- [x] `requirements.lock`
- [ ] Manual Docker smoke (upload + request-id header)

## Out of scope / follow-ups

- Auth / API keys
- OpenTelemetry traces
- Rate limiting / WAF
- Kubernetes manifests
- Centralized log shipping config
