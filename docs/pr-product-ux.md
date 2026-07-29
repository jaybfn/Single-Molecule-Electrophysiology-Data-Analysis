# PR: Product UX (preview, exports, tutorial, progress)

**Branch:** `feature/product-ux` (or current working branch) → `main`  
**Version:** 2.5.0

## Summary

Improves the lab-facing Streamlit experience:

1. **Live threshold / baseline preview** before full detection (`POST /v1/preview`)
2. **Analysis window** — select a time region on the full-trace overview; detect/PSD only that range (`t_start` / `t_end`)
3. **Exports** — events CSV, fit/PSD JSON, Plotly HTML / JSON / PNG
4. **First-analysis tutorial** + one-click **example CSV**
5. **Progress status** panels for detect / dwell / PSD on large uploads

## Motivation

Scientists need to tune thresholds visually, leave with publishable artifacts, and
follow a short happy path — without waiting for a full detect on every slider move.

## What’s changed

### API
- `event-service` + gateway: `POST /v1/preview` → downsampled `time` / `current` + metadata

### Web UI
- Sidebar: example dataset, file size, clear session, live-preview toggle
- Event tab: residual threshold overlay updates with direction/baseline/multipliers
- `st.status` progress for long-running calls
- Download buttons for tables, fit JSON, spectrum CSV, and plots (HTML/JSON/PNG via kaleido)

### Docs / ops
- [docs/first_analysis.md](first_analysis.md)
- Example CSV copied into web-ui Docker image (`EXAMPLE_CSV_PATH`)
- Package **2.5.0**

## How to test

```bash
pip install -e ".[dev,viz,services]"
pytest tests/test_services.py tests/test_serving.py -k "preview or event or health"

docker compose up --build
# UI: Use example CSV → tune thresholds → Run detection → download CSV/HTML
curl -F "file=@data/test.csv" "http://localhost:8000/v1/preview?max_points=5000"
```

## Checklist

- [x] Preview endpoint + gateway proxy
- [x] Live threshold overlay in UI
- [x] CSV / JSON / plot exports
- [x] Example dataset + first-analysis doc
- [x] Progress status for long jobs
- [ ] Manual Docker smoke on a real ABF

## Out of scope / follow-ups

- True mid-request cancellation (Streamlit/httpx limitation)
- Auth / multi-user sessions
- Client-side ABF decode without gateway
- Interactive brush-zoom sync to detection window API
