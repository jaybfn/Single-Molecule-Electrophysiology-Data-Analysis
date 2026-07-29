"""Tests for shared HTTP serving helpers (middleware, errors, limits)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient

from pynanopore.serving import ServiceSettings, configure_service, error_body
from pynanopore.serving.app_factory import enforce_upload_size

ROOT = Path(__file__).resolve().parents[1]


def _load_app(module_name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_error_body_shape():
    body = error_body(code="bad_request", message="nope", request_id="abc", details={"x": 1})
    assert body["error"]["code"] == "bad_request"
    assert body["error"]["request_id"] == "abc"
    assert body["error"]["details"]["x"] == 1


def test_request_id_generated_and_echoed():
    settings = ServiceSettings(service_name="test-svc", log_json=False, max_upload_mb=1)
    app = FastAPI()
    configure_service(app, settings)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID")
    assert resp.headers.get("X-Service") == "test-svc"


def test_request_id_propagated_from_header():
    settings = ServiceSettings(service_name="test-svc", log_json=False)
    app = FastAPI()
    configure_service(app, settings)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/ping", headers={"X-Request-ID": "fixed-id-123"})
    assert resp.headers.get("X-Request-ID") == "fixed-id-123"


def test_content_length_too_large_returns_413():
    settings = ServiceSettings(service_name="test-svc", log_json=False, max_upload_mb=1)
    app = FastAPI()
    configure_service(app, settings)

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        return {"name": file.filename}

    client = TestClient(app)
    # Claim a body larger than 1 MB via Content-Length without sending that much
    resp = client.post(
        "/upload",
        content=b"tiny",
        headers={
            "Content-Length": str(2 * 1024 * 1024),
            "Content-Type": "application/octet-stream",
        },
    )
    assert resp.status_code == 413
    body = resp.json()
    assert body["error"]["code"] == "upload_too_large"
    assert body["error"]["request_id"]


def test_enforce_upload_size_raises():
    settings = ServiceSettings(max_upload_mb=1)
    with pytest.raises(Exception) as excinfo:
        enforce_upload_size(b"x" * (1024 * 1024 + 1), settings, "rid")
    assert excinfo.value.status_code == 413


def test_validation_error_envelope():
    settings = ServiceSettings(service_name="test-svc", log_json=False)
    app = FastAPI()
    configure_service(app, settings)

    from pydantic import BaseModel, Field

    class Body(BaseModel):
        n: int = Field(..., ge=1)

    @app.post("/n")
    def post_n(body: Body):
        return body

    client = TestClient(app)
    resp = client.post("/n", json={"n": 0})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert "errors" in body["error"]["details"]


def test_event_service_returns_request_id_header():
    mod = _load_app("event_svc_app_hardening", ROOT / "services" / "event-service" / "app.py")
    client = TestClient(mod.app)
    resp = client.get("/health", headers={"X-Request-ID": "evt-1"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == "evt-1"


def test_stats_bad_request_envelope():
    mod = _load_app("stats_svc_app_hardening", ROOT / "services" / "stats-service" / "app.py")
    client = TestClient(mod.app)
    resp = client.post("/v1/dwelltime", json={"events": []})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "bad_request"
    assert body["error"]["request_id"]
