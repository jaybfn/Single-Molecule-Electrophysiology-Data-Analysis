# PR: Add MLE dwell-time fitting and wire pulse-shape detection into the API/UI

**Branch:** `feature/dwelltime` → `main`  
**Version:** 2.2.0

## Summary

This PR advances scientific analysis depth in two complementary ways:

1. **Dwell-time lifetimes** — proper MLE exponential / bi-exponential fits with AIC/BIC model selection (replacing reliance on unconstrained histogram `a·e^{bx}` as the primary path).
2. **End-to-end UX** — expose Phase A detection controls (direction, baseline, overlap, pulse-shape idealization) through **event-service → gateway → Streamlit UI**, so Docker users can exercise the screenshot-style overlay.

## Motivation

- Histogram curve-fits do not yield a clean physical \(\tau\); MLE on unbinned dwells does (\(\hat\tau=\bar t\) for a single exponential).
- Dual-component kinetics need a mixture model plus AIC/BIC to choose single vs double.
- Pulse-shape / polarity / baseline lived only in the core library; the Compose UI could not configure or display them.

## What’s changed

### Core (`pynanopore` 2.2.0)

- `DwellTimeExponentialFit.fit(method="mle"|"histogram", fit_type="single"|"double"|"auto")`
- `DwellTimeFitResult` with `parameters`, `log_likelihood`, `aic`, `bic`, histogram + fitted density
- Log or linear binning for display
- `compare_models()` for single vs double AIC/BIC
- Legacy `fit_data` / `a·e^{bx}` retained for compatibility
- CLI: `pynanopore dwelltime ... --method mle --fit auto --binning log`

### Services & UI

- **event-service**: `direction`, `baseline`, `baseline_window`, `overlap`, `include_pulse_plot` → returns `pulse_plot`
- **gateway**: proxies the new detect/dwelltime fields
- **stats-service**: `method`, `binning`, `fit_type=auto`; response includes AIC/BIC and optional model comparison
- **web-ui**: sidebar controls for detection + pulse overlay; stats tab for MLE/auto/log bins

### Docs & tests

- [docs/dwelltime_math.md](../docs/dwelltime_math.md) — MLE, mixture, AIC/BIC
- README links to dwell-time math
- New `tests/test_dwelltime_mle.py`; service contract updated for MLE `tau`

## How to test

```bash
pip install -e ".[dev,viz,services]"
pytest tests/test_dwelltime_mle.py tests/test_services.py tests/test_dwelltime.py

docker compose up --build
# UI: http://localhost:8501
# Set direction=up, baseline=median, enable pulse shape → Run detection
# Stats: method=mle, fit=auto → Fit dwell times
```

```bash
pynanopore dwelltime events.csv --fit auto --method mle --binning log
```

## Checklist

- [x] MLE recovers planted single-\(\tau\) on synthetic data
- [x] Auto model selection + AIC/BIC exposed in API/UI
- [x] Pulse-shape plot returned from detect path
- [x] Unit + service tests green
- [ ] Manual check on a real ABF in Docker UI

## Out of scope / follow-ups

- PSD composite / \(1/f\) models (Phase C)
- Batch multi-file CLI pipelines (Phase D)
- Auth, rate limits, structured request IDs
