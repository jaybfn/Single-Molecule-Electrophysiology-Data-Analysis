# Hosted demo (optional CD)

Run a public (or lab-VPN) UI without asking users to clone the repo.

**Default recommendation:** one small VM + Docker Compose + Caddy.  
Fly.io / Railway work too, but five interdependent services are simpler on Compose.

## Security baseline (do this before opening the internet)

| Control | Setting |
|---------|---------|
| Upload cap | `MAX_UPLOAD_MB=25`–`50` (compose default in prod file is 50) |
| Timeouts | `HTTP_TIMEOUT_S=120`, `DOWNSTREAM_TIMEOUT_S=120` |
| Public ports | Prefer **only** 80/443 via Caddy; do not publish 8001–8003 |
| Auth | Put basic auth / SSO / VPN in front of the UI for anything beyond a throwaway demo |
| Pin images | `PYNANOPORE_TAG=v2.7.1` (never rely on `:latest` alone for a durable demo) |

## Path A — VM + Compose + Caddy

1. Provision Ubuntu 22.04+ (2 vCPU / 4 GB RAM is a reasonable start).
2. Install Docker Engine + Compose plugin + Caddy.
3. Copy or clone this repo (or only `compose.prod.yml` + `deploy/Caddyfile`).
4. Pull and start a **pinned** release:

```bash
export PYNANOPORE_TAG=v2.7.1
export MAX_UPLOAD_MB=50
# GHCR (default):
docker compose -f compose.prod.yml pull
docker compose -f compose.prod.yml up -d

# Or Docker Hub:
# export IMAGE_PREFIX=yourdockerhubuser
# docker compose -f compose.prod.yml pull && docker compose -f compose.prod.yml up -d
```

5. Point DNS at the VM and run Caddy (edit `deploy/Caddyfile`, set `DEMO_DOMAIN` / `CADDY_ACME_EMAIL`):

```bash
export DEMO_DOMAIN=demo.example.com
export CADDY_ACME_EMAIL=you@example.com
caddy run --config deploy/Caddyfile
```

6. Open `https://demo.example.com` (Streamlit). Keep gateway ports firewalled.

### Updating the demo on a new release

```bash
export PYNANOPORE_TAG=v2.7.2   # new tag after Release workflow finishes
docker compose -f compose.prod.yml pull
docker compose -f compose.prod.yml up -d
```

Optional CD: a tag-triggered SSH/`docker compose` Action using a deploy key — only enable with IP allowlists or auth on the UI.

## Path B — Fly.io / Railway (sketch)

- Treat each service as its own app/service, or use a platform that runs full Compose.
- Wire private networking so only `web-ui` (and maybe `gateway`) is public.
- Set the same env vars as `compose.prod.yml`.
- Deploy **only on `v*` tags** after the Release workflow has pushed images.

Exact Fly/Railway manifests drift often; start from Path A unless you already standardize on one PaaS.

## Release workflow hook

Images are published by [`.github/workflows/release.yml`](../.github/workflows/release.yml) to:

- `ghcr.io/jaybfn/pynanopore-*:vX.Y.Z` (always, via `GITHUB_TOKEN`)
- `DOCKER_USERNAME/pynanopore-*:vX.Y.Z` (optional Hub secrets)

After the first GHCR push, make packages **public** under GitHub → Packages (or keep private and `docker login ghcr.io` on the VM).
