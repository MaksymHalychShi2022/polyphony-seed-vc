import os
from pathlib import Path

import click
import torch
from tqdm import tqdm

from seed_vc.features.f0 import F0FeatureExtractor
from seed_vc.train.features_dataset import load_source_target_pairs


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@click.command(context_settings={"show_default": True})
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Device used by RMVPE f0 extractor.",
)
def main(device: str) -> None:
    data_processed = Path(os.environ["DATA_PROCESSED"])
    features_root = Path(os.environ["DATA_FEATURES"])

    extractor = F0FeatureExtractor(
        features_root=features_root,
        device=_resolve_device(device),
        require_features=False,
    )

    all_pairs = []
    for split in ("train", "val"):
        csv_path = data_processed / f"{split}.csv"
        if csv_path.exists():
            all_pairs.extend(load_source_target_pairs(csv_path))

    unique_audio_paths = sorted(
        {src for src, _ in all_pairs} | {tgt for _, tgt in all_pairs}
    )

    features_before = sum(
        1 for p in unique_audio_paths if extractor.get_feature_path(p).exists()
    )

    for audio_path in tqdm(unique_audio_paths, desc="Precomputing f0", unit="audio"):
        _ = extractor.extract(audio_path)

    features_after = sum(
        1 for p in unique_audio_paths if extractor.get_feature_path(p).exists()
    )

    click.echo(
        "Done. "
        f"Unique audio files: {len(unique_audio_paths)}. "
        f"Features before: {features_before}. "
        f"Features after: {features_after}. "
        f"Written: {max(0, features_after - features_before)}."
    )


if __name__ == "__main__":
    main()
