import csv
from pathlib import Path

import click
import numpy as np
import torch
import yaml
from tqdm import tqdm

from seed_vc.features.mel import MelSpectrogramExtractor
from seed_vc.features.semantic import WhisperFeatureExtractor


def load_source_target_pairs(data_path: str | Path) -> list[tuple[Path, Path]]:
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
    csv_path = Path(data_path).expanduser().resolve()
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(
            "Only CSV input is supported. Provide a CSV with source,target columns."
        )

    pairs = []
    base_dir = csv_path.parent
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV is empty.")

        fieldnames = [name.strip().lower() for name in reader.fieldnames if name]
        if "source" not in fieldnames or "target" not in fieldnames:
            raise ValueError("CSV must include header columns: source,target")

        for row in reader:
            normalized_row = {
                (k.strip().lower() if k else ""): (
                    v.strip() if isinstance(v, str) else v
                )
                for k, v in row.items()
            }
            src_rel = normalized_row.get("source")
            tgt_rel = normalized_row.get("target")
            if not src_rel or not tgt_rel:
                continue

            src_path = (base_dir / src_rel).expanduser().resolve()
            tgt_path = (base_dir / tgt_rel).expanduser().resolve()
            if not src_path.exists() or not tgt_path.exists():
                continue
            if (
                src_path.suffix.lower() not in audio_exts
                or tgt_path.suffix.lower() not in audio_exts
            ):
                continue
            pairs.append((src_path, tgt_path))

    if len(pairs) == 0:
        raise ValueError("CSV provided but no valid (source,target) pairs found.")
    return pairs


class TargetSourcePairsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str | Path, batch_size: int = 1):
        self.data_path = data_path
        self.data = load_source_target_pairs(data_path)

        assert len(self.data) != 0
        while len(self.data) < batch_size:
            self.data += self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        src_path, tgt_path = self.data[idx]
        return src_path, tgt_path


class FeaturesDataset(TargetSourcePairsDataset):
    def __init__(
        self,
        data_path: str | Path,
        spect_params: dict,
        whisper_model_name: str,
        cache_root: str | Path,
        sr: int = 22050,
        batch_size: int = 1,
        require_cache: bool = True,
        semantic_device: str = "cpu",
    ):
        super().__init__(data_path=data_path, batch_size=batch_size)
        self.require_cache = require_cache
        self.mel_extractor = MelSpectrogramExtractor(
            spect_params=spect_params,
            sr=sr,
            cache_root=cache_root,
            require_cache=require_cache,
        )
        self.semantic_extractor = WhisperFeatureExtractor(
            whisper_model_name=whisper_model_name,
            cache_root=cache_root,
            device=semantic_device,
            require_cache=require_cache,
        )

    def __getitem__(self, idx):
        src_path, tgt_path = super().__getitem__(idx)
        src_mel = self.mel_extractor.extract(src_path, require_cache=self.require_cache)
        tgt_mel = self.mel_extractor.extract(tgt_path, require_cache=self.require_cache)
        src_semantic = self.semantic_extractor.extract(
            src_path, require_cache=self.require_cache
        )
        tgt_semantic = self.semantic_extractor.extract(
            tgt_path, require_cache=self.require_cache
        )
        return (
            src_mel,
            tgt_mel,
            src_semantic,
            tgt_semantic,
            str(src_path),
            str(tgt_path),
        )


def collate_features(batch):
    batch_size = len(batch)
    lengths = [b[1].shape[1] for b in batch]
    batch_indexes = np.argsort(lengths)[::-1]
    batch = [batch[bid] for bid in batch_indexes]

    n_mels = batch[0][0].size(0)
    max_src_mel_len = max([b[0].shape[1] for b in batch])
    max_tgt_mel_len = max([b[1].shape[1] for b in batch])

    src_mels = torch.zeros((batch_size, n_mels, max_src_mel_len)).float() - 10
    tgt_mels = torch.zeros((batch_size, n_mels, max_tgt_mel_len)).float() - 10
    src_mel_lengths = torch.zeros(batch_size).long()
    tgt_mel_lengths = torch.zeros(batch_size).long()

    semantic_dim = batch[0][2].size(1)
    max_src_semantic_len = max([b[2].shape[0] for b in batch])
    max_tgt_semantic_len = max([b[3].shape[0] for b in batch])
    src_semantics = torch.zeros(
        (batch_size, max_src_semantic_len, semantic_dim)
    ).float()
    tgt_semantics = torch.zeros(
        (batch_size, max_tgt_semantic_len, semantic_dim)
    ).float()
    src_semantic_lengths = torch.zeros(batch_size).long()
    tgt_semantic_lengths = torch.zeros(batch_size).long()

    src_paths = []
    tgt_paths = []
    for bid, (
        src_mel,
        tgt_mel,
        src_semantic,
        tgt_semantic,
        src_path,
        tgt_path,
    ) in enumerate(batch):
        src_mel_size = src_mel.size(1)
        tgt_mel_size = tgt_mel.size(1)
        src_semantic_size = src_semantic.size(0)
        tgt_semantic_size = tgt_semantic.size(0)
        src_mels[bid, :, :src_mel_size] = src_mel
        tgt_mels[bid, :, :tgt_mel_size] = tgt_mel
        src_semantics[bid, :src_semantic_size, :] = src_semantic
        tgt_semantics[bid, :tgt_semantic_size, :] = tgt_semantic
        src_mel_lengths[bid] = src_mel_size
        tgt_mel_lengths[bid] = tgt_mel_size
        src_semantic_lengths[bid] = src_semantic_size
        tgt_semantic_lengths[bid] = tgt_semantic_size
        src_paths.append(src_path)
        tgt_paths.append(tgt_path)

    return (
        src_mels,
        src_mel_lengths,
        tgt_mels,
        tgt_mel_lengths,
        src_semantics,
        src_semantic_lengths,
        tgt_semantics,
        tgt_semantic_lengths,
        src_paths,
        tgt_paths,
    )


def build_features_dataloader(
    data_path: str | Path,
    spect_params: dict,
    whisper_model_name: str,
    cache_root: str | Path,
    sr: int,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    require_cache: bool = True,
    semantic_device: str = "cpu",
):
    dataset = FeaturesDataset(
        data_path=data_path,
        spect_params=spect_params,
        whisper_model_name=whisper_model_name,
        cache_root=cache_root,
        sr=sr,
        batch_size=batch_size,
        require_cache=require_cache,
        semantic_device=semantic_device,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_features,
    )


@click.command(help="Iterate FeaturesDataset once and exit.")
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
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Config used to load preprocess_params.sr and spect_params.",
)
@click.option(
    "--batch-size",
    default=1,
    show_default=True,
    type=int,
    help="Batch size.",
)
@click.option(
    "--cache-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Cache root. Features are stored under <cache-root>/features/.",
)
@click.option(
    "--require-cache/--allow-compute-missing",
    default=True,
    show_default=True,
    help=(
        "When enabled, fail if cache is missing. "
        "Disable to compute-and-cache missing mels and semantics while iterating."
    ),
)
@click.option(
    "--semantic-device",
    default="cpu",
    show_default=True,
    type=click.Choice(["cpu", "cuda"]),
    help="Device used for semantic feature extraction when cache is missing.",
)
def main(
    dataset_path: Path,
    config_path: Path,
    batch_size: int,
    cache_root: Path,
    require_cache: bool,
    semantic_device: str,
):
    config = yaml.safe_load(config_path.read_text())
    preprocess_params = config["preprocess_params"]
    sr = int(preprocess_params.get("sr", 22050))
    spect_params = preprocess_params["spect_params"]
    speech_tokenizer = config["model_params"]["speech_tokenizer"]
    speech_tokenizer_type = speech_tokenizer.get("type", "cosyvoice")
    if speech_tokenizer_type != "whisper":
        raise ValueError(
            f"Unsupported speech tokenizer type: {speech_tokenizer_type}. Expected 'whisper'."
        )
    whisper_model_name = speech_tokenizer["name"]

    dataloader = build_features_dataloader(
        data_path=str(dataset_path),
        spect_params=spect_params,
        whisper_model_name=whisper_model_name,
        cache_root=cache_root,
        sr=sr,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        require_cache=require_cache,
        semantic_device=semantic_device,
    )

    first_logged = False
    for batch in tqdm(dataloader, desc="Processing", unit=" batch"):
        (
            src_mels,
            src_mel_lengths,
            tgt_mels,
            tgt_mel_lengths,
            src_semantics,
            src_semantic_lengths,
            tgt_semantics,
            tgt_semantic_lengths,
            src_paths,
            tgt_paths,
        ) = batch
        if not first_logged:
            print("First batch details:")
            print(f"  src_mels shape: {tuple(src_mels.shape)}")
            print(f"  tgt_mels shape: {tuple(tgt_mels.shape)}")
            print(f"  src_mel_lengths: {src_mel_lengths.tolist()}")
            print(f"  tgt_mel_lengths: {tgt_mel_lengths.tolist()}")
            print(f"  src_semantics shape: {tuple(src_semantics.shape)}")
            print(f"  tgt_semantics shape: {tuple(tgt_semantics.shape)}")
            print(f"  src_semantic_lengths: {src_semantic_lengths.tolist()}")
            print(f"  tgt_semantic_lengths: {tgt_semantic_lengths.tolist()}")
            print(f"  src_paths[0]: {src_paths[0] if src_paths else 'N/A'}")
            print(f"  tgt_paths[0]: {tgt_paths[0] if tgt_paths else 'N/A'}")
            first_logged = True

    print("Done.")


if __name__ == "__main__":
    main()
