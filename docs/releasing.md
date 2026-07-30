# Releasing pynanopore

Checklist for shipping a versioned release (`vX.Y.Z`).

## Before tagging

1. **Version** — bump in lockstep:
   - `packages/pynanopore/src/pynanopore/_version.py`
   - `pyproject.toml` (`[project].version`)
   - FastAPI `version=` on gateway + analysis services
2. **CHANGELOG.md** — add a section for the new version (what users care about).
3. **CI green on `main`** — lint, mypy, tests, wheel build.
4. **Publish credentials** (one-time):
   - **PyPI Trusted Publishing** — project `pynanopore` trusts this GitHub repo + workflow `CI` / environment (or default) for the `publish` job (`id-token: write`).
   - **Docker Hub** — repository secrets `DOCKER_USERNAME` and `DOCKER_PASSWORD` (or access token).

## Tag and push

From a clean `main` (after merging the version bump):

```bash
git tag -a v2.7.0 -m "pynanopore 2.7.0"
git push origin main
git push origin v2.7.0
```

Pushing `v*` runs the CI `publish` (PyPI) and `docker` (Hub) jobs after `quality` passes.

## Verify

- [ ] GitHub Actions: tag workflow green
- [ ] PyPI: https://pypi.org/project/pynanopore/
- [ ] Install: `pip install pynanopore==2.7.0` then `pynanopore --version`
- [ ] Docker Hub images tagged `:v2.7.0` and `:latest` for gateway, event/stats/psd services, web-ui
- [ ] Optional GitHub Release notes from `CHANGELOG.md`

## Optional GitHub Release

```bash
gh release create v2.7.0 --title "pynanopore 2.7.0" --notes-file CHANGELOG.md
```

(Or paste the `## [2.7.0]` section only.)
