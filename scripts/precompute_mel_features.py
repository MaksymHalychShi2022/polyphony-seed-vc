import os
from pathlib import Path

import click
import yaml
from tqdm import tqdm

from seed_vc.features.mel import MelSpectrogramExtractor
from seed_vc.train.features_dataset import load_source_target_pairs


@click.command(context_settings={"show_default": True})
@click.option(
    "--config",
    "config_path",
    default="configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Config used for preprocess_params.sr and spect_params.",
)
def main(config_path: Path) -> None:
    data_processed = Path(os.environ["DATA_PROCESSED"])
    features_root = Path(os.environ["DATA_FEATURES"])

    config = yaml.safe_load(config_path.read_text())
    preprocess_params = config["preprocess_params"]
    sr = int(preprocess_params.get("sr", 22050))
    spect_params = preprocess_params["spect_params"]

    extractor = MelSpectrogramExtractor(
        spect_params=spect_params,
        sr=sr,
        features_root=features_root,
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

    for audio_path in tqdm(unique_audio_paths, desc="Precomputing mel", unit="audio"):
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
