# Changelog

All notable changes to **pynanopore** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.7.0] — 2026-07-30

First public tagged release: library on PyPI, Docker images for all Compose services, and a two-minute try path in the README.

### Added

- Multi-level conductance overlay (`idealize_multilevel`, `plot_multi_level`) and `levels_plot` in detect API responses
- Streamlit tabs (Event / Dwell-time / PSD) moved to the top so secondary analyses are not buried
- `CHANGELOG.md` and [docs/releasing.md](docs/releasing.md) release checklist
- CI: tag pushes (`v*`) run quality, PyPI publish, and Docker Hub push for **all** services (gateway, event, stats, psd, web-ui)

### Changed

- Service OpenAPI versions aligned to **2.7.0**

### Includes (from 2.4–2.6, previously untagged)

- Production hardening (`pynanopore.serving`, healthchecks, upload limits, request IDs)
- Product UX (`/v1/preview`, analysis window, exports, example CSV, first-analysis tutorial)
- Science Phase E: multi-level features, `PercentileBaseline`, multi-Lorentzian PSD, parallel batch (`n_jobs`)

## [2.6.0] — 2026-07

Science Phase E (see [docs/science_phase_e.md](docs/science_phase_e.md)).

## [2.5.0] — 2026-07

Product UX: preview, analysis window, exports, tutorial.

## [2.4.0] — 2026-07

Production hardening: structured logging, healthchecks, lockfile.

[2.7.0]: https://github.com/jaybfn/Single-Molecule-Electrophysiology-Data-Analysis/releases/tag/v2.7.0
