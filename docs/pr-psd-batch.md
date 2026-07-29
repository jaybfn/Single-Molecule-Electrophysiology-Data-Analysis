# PR: Add PSD depth (composite fits) and batch multi-file detection

**Branch:** `feature/psd-batch` (or current working branch) → `main`  
**Version:** 2.3.0

## Summary

Delivers **Phase C** (PSD analysis depth) and **Phase D** (batch pipelines) together:

1. Configurable Welch PSD + Lorentzian diagnostics + optional **Lorentzian + \(1/f^\alpha\)** composite model, wired through API/UI.
2. **`batch-detect`** pipeline for folders of ABF/CSV recordings with per-file events, summary table, and versioned run metadata.

## Motivation

- Lab PSD analysis needs tunable Welch parameters and often a power-law continuum under the Lorentzian corner.
- Real workflows process many files; a reproducible batch output schema is required beyond the interactive UI.

## What’s changed

### Phase C — PSD
- `PSDAnalyzer.compute_psd(...)` with `window`, `scaling`, `nperseg`, `noverlap`, `skip_bins`
- `LorentzianFitter` now reports `diagnostics` (`r2_log`, `rmse_log`, `n_points`)
- New `CompositePSDFitter`: \(S(f)=S_0/(1+(f/f_c)^2)+A/f^{\alpha}\)
- `psd-service` / gateway / web-ui expose fit model + Welch knobs
- Docs: [docs/psd_math.md](psd_math.md)

### Phase D — Batch
- `pynanopore.batch.batch_detect(input_dir, output_dir, config)`
- Outputs: `events/*_events.csv`, `summary.csv`, `run_metadata.json` (`schema_version=1.0.0`)
- Optional per-file dwell MLE summary columns
- CLI: `pynanopore batch-detect ./recordings -o ./results`
- Docs: [docs/batch_analysis.md](batch_analysis.md)

### Version
- Package bumped to **2.3.0** (`_version.py` single source)

## How to test

```bash
pip install -e ".[dev,viz,services]"
pytest tests/test_psd_advanced.py tests/test_batch.py tests/test_powerspectrum.py

pynanopore psd data/test.csv --fit --fit-model composite --window hann
pynanopore batch-detect ./some_folder -o ./results --direction down --baseline median

docker compose up --build
# UI PSD tab: choose composite / window / nperseg
```

## Checklist

- [x] Welch knobs + Lorentzian diagnostics
- [x] Composite Lorentzian+\(1/f\) fit
- [x] Batch detect writes schema metadata + summary
- [x] Unit tests for PSD advanced + batch
- [ ] Manual Docker UI check on a real ABF

## Out of scope / follow-ups

- Auth / multi-tenant
- Kubernetes
- Parallel/distributed batch workers
- Additional PSD models (multi-Lorentzian, white-noise floor term as separate UI toggle)
