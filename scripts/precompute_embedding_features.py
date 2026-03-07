from pathlib import Path

import click
import torch
from tqdm import tqdm

from seed_vc.features.embedding import CampplusEmbeddingExtractor
from seed_vc.train.features_dataset import TargetSourcePairsDataset


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@click.command(context_settings={"show_default": True})
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to CSV with source,target rows.",
)
@click.option(
    "--cache-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Cache root. Features are stored under <cache-root>/features/.",
)
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Device used by campplus embedding extractor.",
)
def main(dataset_path: Path, cache_root: Path, device: str) -> None:
    extractor = CampplusEmbeddingExtractor(
        cache_root=cache_root,
        device=_resolve_device(device),
        require_cache=False,
    )

    pairs_dataset = TargetSourcePairsDataset(dataset_path)
    unique_target_paths = sorted({tgt for _, tgt in pairs_dataset.data})

    cache_files_before = sum(
        1
        for audio_path in unique_target_paths
        if extractor.get_cache_path(audio_path).exists()
    )

    for audio_path in tqdm(
        unique_target_paths,
        desc="Precomputing embedding",
        unit="audio",
    ):
        _ = extractor.extract(audio_path)

    cache_files_after = sum(
        1
        for audio_path in unique_target_paths
        if extractor.get_cache_path(audio_path).exists()
    )
    cache_writes = max(0, cache_files_after - cache_files_before)

    click.echo(
        "Done. "
        f"Pairs: {len(pairs_dataset)}. "
        f"Unique target audio files: {len(unique_target_paths)}. "
        f"Cache files before: {cache_files_before}. "
        f"Cache files after: {cache_files_after}. "
        f"Cache writes: {cache_writes}."
    )


if __name__ == "__main__":
    main()
