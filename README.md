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
