from pathlib import Path

import click
import yaml
from tqdm import tqdm

from seed_vc.features.mel import MelSpectrogramExtractor
from seed_vc.train.features_dataset import TargetSourcePairsDataset


@click.command(context_settings={"show_default": True})
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to CSV with source,target rows.",
)
@click.option(
    "--config",
    "config_path",
    default="configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Config used for preprocess_params.sr and spect_params.",
)
@click.option(
    "--cache-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Cache root. Features are stored under <cache-root>/features/.",
)
def main(dataset_path: Path, config_path: Path, cache_root: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    preprocess_params = config["preprocess_params"]
    sr = int(preprocess_params.get("sr", 22050))
    spect_params = preprocess_params["spect_params"]

    extractor = MelSpectrogramExtractor(
        spect_params=spect_params,
        sr=sr,
        cache_root=cache_root,
        require_cache=False,
    )
    pairs_dataset = TargetSourcePairsDataset(dataset_path)
    unique_audio_paths = sorted(
        {src for src, _ in pairs_dataset.data} | {tgt for _, tgt in pairs_dataset.data}
    )

    cache_files_before = sum(
        1
        for audio_path in unique_audio_paths
        if extractor.get_cache_path(audio_path).exists()
    )

    for audio_path in tqdm(unique_audio_paths, desc="Precomputing mel", unit="audio"):
        _ = extractor.extract(audio_path)

    cache_files_after = sum(
        1
        for audio_path in unique_audio_paths
        if extractor.get_cache_path(audio_path).exists()
    )
    cache_writes = max(0, cache_files_after - cache_files_before)

    click.echo(
        "Done. "
        f"Pairs: {len(pairs_dataset)}. "
        f"Unique audio files: {len(unique_audio_paths)}. "
        f"Cache files before: {cache_files_before}. "
        f"Cache files after: {cache_files_after}. "
        f"Cache writes: {cache_writes}."
    )


if __name__ == "__main__":
    main()
