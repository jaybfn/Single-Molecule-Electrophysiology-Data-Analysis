# First analysis tutorial

This walkthrough gets you from a raw recording to events, dwell-time fits, and PSD
using the Docker Compose stack (or a local gateway).

## 1. Start the stack

```bash
docker compose up --build
```

Open the UI at http://localhost:8501 (gateway docs: http://localhost:8000/docs).

## 2. Load a recording

- Click **Use example CSV** in the sidebar (ships `data/test.csv`), **or**
- Upload your own `.abf` / `.csv` file.

The UI shows file size and loads a **downsampled preview** via `POST /v1/preview`.

## 3. Select a region, then tune thresholds

1. Inspect the **full-trace overview**.
2. Drag the **Analysis window** slider to the region you care about
   (or click **Use full recording**).
3. Only that window is sent to detection / PSD.
4. Tune direction / baseline / multipliers on the live residual preview.

## 4. Run detection

Click **Run event detection**. Progress appears in a status panel
(upload → detect → done). Then:

- Inspect the event table and pulse-shape overlay (if enabled)
- **Download events CSV** / detect JSON / plot HTML·PNG

## 5. Dwell-time statistics

On **Statistical Analysis**:

1. Choose fit type (`single` / `double` / `auto`), method (`mle` preferred), binning
2. **Fit dwell times**
3. Download **fit JSON** and the histogram plot

## 6. Power spectrum

On **Power Spectrum**:

1. Pick Lorentzian or composite \(1/f\) model and Welch options
2. **Compute PSD** (uses detection preview current when available)
3. Download PSD fit JSON, spectrum CSV, and plot files

## CLI alternative

```bash
pip install -e ".[dev,viz]"
pynanopore detect data/test.csv -o events.csv
pynanopore dwelltime events.csv --fit single --method mle
pynanopore psd data/test.csv --fit --fit-model lorentzian
```

## Batch folders

For many files, prefer the batch pipeline:

```bash
pynanopore batch-detect ./recordings -o ./results
```

See [batch_analysis.md](batch_analysis.md).

## Tips for large ABFs

- Narrow the **analysis window** before running detection
- Keep **live preview** on for tuning; only run detect once parameters look right
- Disable pulse-shape idealization to speed up large files
- Raise `HTTP_TIMEOUT_S` / `DOWNSTREAM_TIMEOUT_S` and `MAX_UPLOAD_MB` in Compose if needed
- Use **Clear session** to drop cached results before switching files
