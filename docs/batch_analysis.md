# Batch analysis

`pynanopore.batch.batch_detect` processes every `.abf` / `.csv` in an input folder.

## Outputs

```text
output_dir/
  events/<stem>_events.csv
  summary.csv
  run_metadata.json
```

`run_metadata.json` includes `schema_version` (currently `1.0.0`), package version,
timestamps, and the detector config.

## Summary columns

Per file: `n_events`, `sample_rate`, `duration_s`, `median_dwell`,
`median_delta_i_over_i0`, `median_area`, optional dwell MLE parameters (`dwell_tau`, …).

## CLI

```bash
pynanopore batch-detect ./recordings -o ./results \
  --direction up --baseline median --dwell-fit auto
```
