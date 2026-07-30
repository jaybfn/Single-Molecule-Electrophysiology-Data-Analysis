# Releasing pynanopore

Checklist for shipping a versioned release (`vX.Y.Z`).

## Workflows

| Workflow | Trigger | Jobs |
|----------|---------|------|
| [`ci.yml`](../.github/workflows/ci.yml) | push/PR to `main` | quality (lint/mypy/tests) + `pip-audit` |
| [`release.yml`](../.github/workflows/release.yml) | tag `v*` | quality + audit → **PyPI** (`environment: pypi`) → **GHCR** (+ optional Docker Hub) with SBOM/provenance |

## Before tagging

1. **Version** — bump in lockstep:
   - `packages/pynanopore/src/pynanopore/_version.py`
   - `pyproject.toml` (`[project].version`)
   - FastAPI `version=` on gateway + analysis services
2. **CHANGELOG.md** — add a section for the new version.
3. **CI green on `main`** — both `quality` and `audit`.
4. **One-time publish setup** — below.
5. **Version must match the tag.** Tag `v2.7.1` must ship package `2.7.1`. PyPI never replaces an existing wheel filename.

## One-time setup

### 1. GitHub Environment `pypi`

1. Repo → **Settings → Environments → New environment** → name: `pypi`
2. Add **Required reviewers** and/or a **wait timer** so a bad tag cannot publish alone.
3. No secrets are required for Trusted Publishing (OIDC).

### 2. PyPI Trusted Publisher

Update (or add) the publisher at  
https://pypi.org/manage/project/pynanopore/settings/publishing/

| Field | Value |
|-------|--------|
| PyPI Project Name | `pynanopore` |
| Owner | `jaybfn` |
| Repository name | `Single-Molecule-Electrophysiology-Data-Analysis` |
| Workflow name | `release.yml` (not `ci.yml`) |
| Environment name | `pypi` |

Remove or leave inactive any old publisher that pointed at `ci.yml` with an empty environment.

### 3. Container registries

- **GHCR** — enabled automatically (`packages: write` + `GITHUB_TOKEN`). After the first push, set package visibility (public recommended for `compose.prod.yml` pulls).
- **Docker Hub** (optional) — repo secrets `DOCKER_USERNAME` / `DOCKER_PASSWORD`. If unset, Release still pushes GHCR only.

## Tag and push

```bash
git push origin main
git tag -a v2.7.2 -m "pynanopore 2.7.2"
git push origin v2.7.2
```

Approve the `pypi` environment deployment when prompted, then confirm Release workflow jobs are green.

## Verify

- [ ] Release workflow: quality, audit, publish, docker all green
- [ ] PyPI: `pip install pynanopore==X.Y.Z` → `pynanopore --version`
- [ ] GHCR: `ghcr.io/jaybfn/pynanopore-gateway:vX.Y.Z` (and sibling images)
- [ ] Optional Docker Hub tags
- [ ] `PYNANOPORE_TAG=vX.Y.Z docker compose -f compose.prod.yml pull`

## Prod / demo images

```bash
export PYNANOPORE_TAG=v2.7.1
docker compose -f compose.prod.yml pull
docker compose -f compose.prod.yml up -d
```

Hosted demo notes: [hosted_demo.md](hosted_demo.md).

## Optional GitHub Release

```bash
gh release create v2.7.1 --title "pynanopore 2.7.1" --notes-file CHANGELOG.md
```
