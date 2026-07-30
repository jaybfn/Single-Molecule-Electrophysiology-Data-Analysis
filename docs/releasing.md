# Releasing pynanopore

Checklist for shipping a versioned release (`vX.Y.Z`).

## Before tagging

1. **Version** — bump in lockstep:
   - `packages/pynanopore/src/pynanopore/_version.py`
   - `pyproject.toml` (`[project].version`)
   - FastAPI `version=` on gateway + analysis services
2. **CHANGELOG.md** — add a section for the new version (what users care about).
3. **CI green on `main`** — lint, mypy, tests, wheel build.
4. **Publish credentials** (one-time) — see below.
5. **Docker Hub** — repository secrets `DOCKER_USERNAME` and `DOCKER_PASSWORD` (or access token).

### PyPI Trusted Publishing (required for tag publish)

The `publish` job uses OIDC (`id-token: write`) — **no** `PYPI_API_TOKEN` secret.
You must register this repo as a trusted publisher on the existing
[pynanopore](https://pypi.org/project/pynanopore/) project.

1. Sign in as the PyPI owner of `pynanopore`.
2. Open **Publishing**: https://pypi.org/manage/project/pynanopore/settings/publishing/
3. Under **Add a new publisher**, use **exactly**:

| Field | Value |
|-------|--------|
| PyPI Project Name | `pynanopore` |
| Owner | `jaybfn` |
| Repository name | `Single-Molecule-Electrophysiology-Data-Analysis` |
| Workflow name | `ci.yml` (filename only, not “CI”) |
| Environment name | *(leave empty)* |

4. Save. Claims from a failed run should then match (`environment: MISSING` is expected when Environment is blank).

If you see `invalid-publisher` / “no corresponding publisher”, the form values above do not match yet (wrong workflow name, environment mismatch, or publisher not saved).

After fixing, **re-run** the failed “Publish to PyPI” job on the `v2.7.0` workflow (or delete + re-push the tag).

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
- [ ] PyPI: https://pypi.org/project/pynanopore/ (expect 2.7.0 after this release)
- [ ] Install: `pip install pynanopore==2.7.0` then `pynanopore --version`
- [ ] Docker Hub images tagged `:v2.7.0` and `:latest` for gateway, event/stats/psd services, web-ui
- [ ] Optional GitHub Release notes from `CHANGELOG.md`

## Optional GitHub Release

```bash
gh release create v2.7.0 --title "pynanopore 2.7.0" --notes-file CHANGELOG.md
```

(Or paste the `## [2.7.0]` section only.)
