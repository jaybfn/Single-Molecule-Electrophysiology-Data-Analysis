# PR: Multi-level conductance plot + Streamlit tab visibility

**Branch:** `feature/multievent-plots` → `main`  
**Follows:** Science Phase E (v2.6.0)

## Summary

Makes multi-level event analysis visible in the UI and fixes Streamlit navigation:

1. **Multi-level conductance overlay** on the detected trace (not only table columns)
2. **Streamlit tabs** (Event / Dwell-time / PSD) moved to the top so they are not buried under the region plot

## Motivation

Phase E added multi-level features (`n_levels`, `level1_*`, `level2_*`) but the Streamlit app only showed them in the events table. Lab users need to **see** subconductance structure on the current trace. Separately, the analysis-window overview sat above the tab bar, so dwell-time and PSD tabs looked missing.

## What’s changed

### Core / viz
- `assign_event_levels` / `idealize_multilevel` in `detection/levels.py` — per-sample level codes + stepwise idealization
- `plot_multi_level(...)` — raw trace + black multi-level idealization; orange = level 1, purple = level 2; open-pore / level guides

### API
- `event-service` detect response adds optional `levels_plot` when `analyze_levels` and pulse plotting are enabled

### Web UI
- Tabs rendered first: **Event Detection | Statistical Analysis | Power Spectrum**
- Region selection moved into an expander inside the Event tab
- Shows **Multi-level conductance overlay** when `levels_plot` is present (plus existing pulse-shape plot)

### Tests
- `test_multilevel_idealization_and_plot` in `tests/test_science_phase_e.py`

## How to test

```bash
pip install -e ".[dev,viz,services]"
pytest tests/test_science_phase_e.py -q

docker compose up --build event-service web-ui
# UI: load example CSV → leave multi-level + pulse plot on → Run event detection
# Expect: multilevel overlay plot, then pulse-shape plot
# Confirm dwell / PSD tabs visible at top without scrolling past the overview
```

## Checklist

- [x] Multi-level idealization + Plotly overlay
- [x] `levels_plot` on detect response
- [x] Streamlit tabs visible at top
- [x] Unit test for idealization/plot
- [ ] Manual Docker smoke on a real multi-level ABF

## Out of scope / follow-ups

- HMM / QuB kinetic idealization
- Brush-select synced to analysis window
- Per-level dwell-time histograms
