# Seed VC

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Eval%20Report-blue?logo=github)](https://maksymhalychshi2022.github.io/polyphony-seed-vc-evaluation/)

## Start demo app

```bash
task demo
```

## Start training

```bash
task train
```

## Prepare dataset for fine-tuning

Provide a CSV that lists explicit source/target pairs (relative to the CSV location):

```csv
source,target
speakerA/utt1.wav,common_target.wav
speakerB/utt2.wav,common_target.wav
```

- Supported extensions: `wav`, `mp3`, `flac`, `ogg`, `m4a`, `opus`.
- Audio length must be between 1s and 30s; anything outside is skipped.
- The CSV path is passed as `--train-dataset`; e.g. `python -m seed_vc.train.train --train-dataset data/pairs.csv --run-name my_run`.

### Training logger contract

- `Trainer` now receives an injected `TrainLogger` instance (constructed in `seed_vc.train.train.main`).
- Text logs from training are mirrored to terminal and `<log_dir>/<run_name>/train.log`.
- Metrics emitted from trainer go through `TrainLogger.metric(...)` and are always written to TensorBoard.
- Remote MLflow logging is enabled only when `MLFLOW_TRACKING_URI` is set; otherwise MLflow is skipped entirely.
- When MLflow is enabled, the logger uses the existing run folder name (`run_name`) as the MLflow experiment and run name.
- Progress bar lifecycle is managed by `TrainLogger.progress(...)`.

### Preprocess dataset

The preprocessing pipeline is managed by DVC. Raw data is tracked manually; processed
data and feature caches are reproduced on demand.

**First-time setup** — pull raw data from DVC remote:

```bash
dvc pull data/raw.dvc
```

**Run the full pipeline** (process audio → build CSVs → extract all features):

```bash
dvc repro
```

DVC skips stages whose inputs and params haven't changed. To force a full rerun:

```bash
dvc repro --force
```

**Tune the number of songs to process** by editing `params.yaml`:

```yaml
process_raw:
  max_songs: 10   # set to -1 for all songs
```

Then `dvc repro` will re-run only the affected stages.

**Run a single stage** (e.g. just re-extract F0 features):

```bash
dvc repro extract-f0
```

**Push outputs to remote** after a successful run:

```bash
dvc push
```

Evaluation summaries now combine several views of quality:
- `resemblyzer_similarity` for target-vs-generated timbre similarity
- `f0_rmse` for source-vs-generated melody error
- `f0_correlation` for source-vs-generated melody contour similarity
- `singmos_naturalness` for no-reference singing naturalness

## Staged evaluation workflow

The evaluation command now supports resumable stages:

```bash
uv run python -m eval.cli --stage generate-results --split val
uv run python -m eval.cli --stage compute-metrics --split val
uv run python -m eval.cli --stage build-report --split val
```

Run all stages end-to-end:

```bash
uv run python -m eval.cli --stage all --split val
```

Default artifacts are written under `.eval_cache/<timestamp>/`:

- `results_manifest.json`
- `metrics_manifest.json`
- `evaluation_report.html`

`metrics_manifest.json` now stores per-item metric values, per-metric statuses, and aggregate summaries for the supported evaluation metrics. The HTML report lets you switch the metric used for sorting/filtering while still auditioning source, target, and generated audio.

SingMOS-Pro scoring is enabled by default during metric computation. Disable it only if you intentionally want to skip the naturalness metric:

```bash
uv run python -m eval.cli --stage compute-metrics --split val --disable-singmos
```
