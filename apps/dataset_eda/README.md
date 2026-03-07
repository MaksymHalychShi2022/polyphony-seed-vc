# Dataset EDA (Streamlit)

Simple Streamlit app to browse `data/raw/videos.json`.

## Run

```bash
# one-off (doesn't touch pyproject.toml)
uv run --with streamlit streamlit run apps/dataset_eda/app.py

# or: add to this repo
uv add streamlit
uv run streamlit run apps/dataset_eda/app.py
```

Or via go-task:

```bash
task dataset-eda
```

## Notes

- Selecting a record shows its details and:
  - audio files discovered under `data/raw/train/<videoId>/` and `data/raw/test/<videoId>/`
  - any audio paths found in the JSON (including `local_mp3_path`)
- Local audio paths are resolved relative to the repo root.
