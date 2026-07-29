"""API contract tests for FastAPI microservices (in-process)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _load_app(module_name: str, path: Path):
    # Ensure service directory is import-safe and avoid name collisions
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def event_client():
    mod = _load_app("event_svc_app", ROOT / "services" / "event-service" / "app.py")
    return TestClient(mod.app)


@pytest.fixture(scope="module")
def stats_client():
    mod = _load_app("stats_svc_app", ROOT / "services" / "stats-service" / "app.py")
    return TestClient(mod.app)


@pytest.fixture(scope="module")
def psd_client():
    mod = _load_app("psd_svc_app", ROOT / "services" / "psd-service" / "app.py")
    return TestClient(mod.app)


def test_event_health(event_client: TestClient):
    resp = event_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_event_detect_csv(event_client: TestClient, csv_trace_path: Path):
    with csv_trace_path.open("rb") as fh:
        resp = event_client.post(
            "/v1/detect",
            files={"file": ("trace.csv", fh, "text/csv")},
            params={
                "std_multiplier": 0.5,
                "threshold_multiplier": 2.0,
                "interval_length": 1.0,
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "events" in body
    assert body["sample_rate"] > 0


def test_stats_dwelltime(stats_client: TestClient):
    rng = np.random.default_rng(0)
    events = [
        {
            "difference": float(x),
            "start_time": 0.0,
            "end_time": float(x),
            "amplitude": 1.0,
        }
        for x in rng.exponential(0.01, size=200)
    ]
    resp = stats_client.post(
        "/v1/dwelltime",
        json={"events": events, "bins": 30, "fit_type": "single", "include_plot": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "a" in body["parameters"]
    assert "tau" in body["parameters"]


def test_psd_array(psd_client: TestClient):
    rng = np.random.default_rng(0)
    current = rng.normal(size=2000).tolist()
    resp = psd_client.post(
        "/v1/psd",
        json={"current": current, "fs": 1000.0, "fit": True, "include_plot": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_frequencies"] > 0
    assert body["S0"] is not None
