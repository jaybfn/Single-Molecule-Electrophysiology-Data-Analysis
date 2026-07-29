# PR: Science Phase E (levels, drift baseline, multi-Lorentzian, parallel batch)

**Branch:** `feature/science-phase-e` (or current) → `main`  
**Version:** 2.6.0

## Summary

Raises analysis quality with four science upgrades:

1. **Multi-level conductance** features inside detected events (1–2 levels via 1-D k-means + BIC)
2. **Percentile open-pore baseline** for long drift / high event occupancy
3. **PSD models**: Lorentzian+white floor and double-Lorentzian (+white)
4. **Parallel batch** workers (`n_jobs`) for multi-file folders

## Motivation

Real recordings often show subconductance states, baseline drift under high occupancy,
multi-corner noise spectra, and dozens of files per experiment — all poorly served by
single-level / single-Lorentzian / serial-only tooling.

## What’s changed

### Core
- `PercentileBaseline(percentile, window_s)`
- `analyze_event_levels` + Event fields (`n_levels`, `level1_*`, `level2_*`, …)
- `LorentzianWhiteFitter`, `MultiLorentzianFitter`
- `BatchDetectConfig.n_jobs` / `baseline='percentile'` / `analyze_levels`
- Batch schema **1.1.0**
- Math notes: [docs/science_phase_e.md](science_phase_e.md)

### API / UI / CLI
- Detect: `baseline=percentile`, `baseline_percentile`, `analyze_levels`
- PSD: `lorentzian_white`, `double_lorentzian` (+ `N`, `S0_2`, `fc_2` in response)
- CLI: matching flags; `batch-detect --n-jobs`
- Streamlit: percentile baseline + new PSD models + multilevel toggle

### Tests
- `tests/test_science_phase_e.py`

## How to test

```bash
pip install -e ".[dev,viz,services]"
pytest tests/test_science_phase_e.py tests/test_event_features.py tests/test_psd_advanced.py tests/test_batch.py

pynanopore detect data/test.csv --baseline percentile --baseline-percentile 90 -o events.csv
pynanopore psd data/test.csv --fit --fit-model double_lorentzian
pynanopore batch-detect ./recordings -o ./results --n-jobs 4 --baseline percentile
```

## Checklist

- [x] Percentile baseline
- [x] Multi-level event features
- [x] Lorentzian+white and double-Lorentzian PSD
- [x] Parallel batch
- [x] CLI / services / UI wiring
- [x] Unit tests + math doc
- [ ] Manual Docker smoke on a multi-level ABF

## Out of scope / follow-ups

- Full HMM / QuB idealization
- >2 conductance levels with dwell-per-level kinetics
- Adaptive number of Lorentzians via AIC across 1–3
- Distributed batch (Ray / cluster)
