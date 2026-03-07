from pathlib import Path

import click
import torch
import yaml
from tqdm import tqdm

from seed_vc.features.semantic import WhisperFeatureExtractor
from seed_vc.train.features_dataset import load_source_target_pairs


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
    "--config",
    "config_path",
    default="configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Config used for model_params.speech_tokenizer settings.",
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
    help="Device used by Whisper semantic extractor.",
)
def main(dataset_path: Path, config_path: Path, cache_root: Path, device: str) -> None:
    config = yaml.safe_load(config_path.read_text())
    speech_tokenizer = config["model_params"]["speech_tokenizer"]
    speech_tokenizer_type = speech_tokenizer.get("type", "cosyvoice")
    if speech_tokenizer_type != "whisper":
        raise ValueError(
            f"Unsupported speech tokenizer type: {speech_tokenizer_type}. Expected 'whisper'."
        )

    whisper_model_name = speech_tokenizer["name"]
    extractor = WhisperFeatureExtractor(
        whisper_model_name=whisper_model_name,
        cache_root=cache_root,
        device=_resolve_device(device),
        require_cache=False,
    )

    pairs = load_source_target_pairs(dataset_path)
    unique_audio_paths = sorted({src for src, _ in pairs} | {tgt for _, tgt in pairs})

    cache_files_before = sum(
        1
        for audio_path in unique_audio_paths
        if extractor.get_cache_path(audio_path).exists()
    )

    for audio_path in tqdm(
        unique_audio_paths, desc="Precomputing semantic", unit="audio"
    ):
        _ = extractor.extract(audio_path)

    cache_files_after = sum(
        1
        for audio_path in unique_audio_paths
        if extractor.get_cache_path(audio_path).exists()
    )
    cache_writes = max(0, cache_files_after - cache_files_before)

    click.echo(
        "Done. "
        f"Pairs: {len(pairs)}. "
        f"Unique audio files: {len(unique_audio_paths)}. "
        f"Cache files before: {cache_files_before}. "
        f"Cache files after: {cache_files_after}. "
        f"Cache writes: {cache_writes}."
    )


if __name__ == "__main__":
    main()
