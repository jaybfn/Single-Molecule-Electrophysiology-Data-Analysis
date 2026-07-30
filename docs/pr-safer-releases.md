# PR: Safer releases + prod image ergonomics

**Branch:** `feature/prod-ready` → `main`

## Summary

Hardens release and deploy infrastructure without changing science APIs:

- Split **CI** (`ci.yml` on push/PR) from **Release** (`release.yml` on `v*` tags) so failures are easier to read
- Gate PyPI publish on GitHub Environment **`pypi`** (reviewers / wait timer)
- Add **`pip-audit`** against `requirements.lock` on CI and Release
- Push images to **GHCR** (optional Docker Hub) with **SBOM + provenance** attestations
- Add **`compose.prod.yml`** to pull pinned tags (`PYNANOPORE_TAG=vX.Y.Z`)
- Document hosted demo path: `deploy/Caddyfile` + `docs/hosted_demo.md`

## Operator follow-ups (do once after merge, before next tag)

1. Create GitHub Environment **`pypi`** (required reviewers and/or wait timer)
2. Update PyPI Trusted Publisher:
   - Owner: `jaybfn`
   - Repository: `Single-Molecule-Electrophysiology-Data-Analysis`
   - Workflow: **`release.yml`**
   - Environment: **`pypi`**
3. After the first GHCR push, set package visibility (public if using unauthenticated `compose.prod.yml` pulls)

## Test plan

- [ ] Push/PR to `main` runs **CI only** (quality + audit) — no PyPI/Docker jobs
- [ ] Tag `v*` runs **Release**; publish waits for Environment approval
- [ ] PyPI publish succeeds with Trusted Publisher (`release.yml` + `pypi`)
- [ ] Images appear at `ghcr.io/jaybfn/pynanopore-*` (`:latest` and `:vX.Y.Z`)
- [ ] `PYNANOPORE_TAG=vX.Y.Z docker compose -f compose.prod.yml pull` works
- [ ] Optional: Docker Hub tags when `DOCKER_USERNAME` / `DOCKER_PASSWORD` are set
