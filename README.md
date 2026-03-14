# Seed VC

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
- The CSV path itself is passed as `--dataset-dir`; e.g. `python -m seed_vc.train.train --dataset-dir data/pairs.csv`.

### Building pairs from the Polyphony Project (multitrack HF dataset)

Pull multitrack songs from Hugging Face, mix them, and emit per-chunk `(source,target)` rows for `FT_Dataset`:

```bash
task prepare-dataset
```

Evaluation results on base checkpoint:
Resemblyzer cosine similarity -> mean: 0.7514, median: 0.7556, std: 0.0400, min: 0.6661, max: 0.8223

Evaluation results on finetuned checkpoint:
Resemblyzer cosine similarity -> mean: 0.8342, median: 0.8342, std: 0.0259, min: 0.7490, max: 0.8727

## Staged evaluation workflow

The evaluation command now supports resumable stages:

```bash
uv run python -m eval.cli --stage generate-results --dataset data/val.csv
uv run python -m eval.cli --stage compute-metrics --dataset data/val.csv
uv run python -m eval.cli --stage build-report --dataset data/val.csv
```

Run all stages end-to-end:

```bash
uv run python -m eval.cli --stage all --dataset data/val.csv
```

Default artifacts are written under `.eval_cache/<timestamp>/`:
- `results_manifest.json`
- `metrics_manifest.json`
- `evaluation_report.html`
