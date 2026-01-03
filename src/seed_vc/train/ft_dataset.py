import random
from pathlib import Path
import csv

import librosa
import numpy as np
import torch

from seed_vc.modules.audio import mel_spectrogram


duration_setting = {
    "min": 1.0,
    "max": 30.0,
}


# assume single speaker
def to_mel_fn(wave, mel_fn_args):
    return mel_spectrogram(wave, **mel_fn_args)


class FT_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        spect_params,
        sr=22050,
        batch_size=1,
        split="train",
    ):
        self.data_path = data_path
        self.split = None if split is None else split.lower()
        self.data = self._build_pairs(data_path)

        self.sr = sr
        self.mel_fn_args = {
            "n_fft": spect_params["n_fft"],
            "win_size": spect_params.get(
                "win_length", spect_params.get("win_size", 1024)
            ),
            "hop_size": spect_params.get(
                "hop_length", spect_params.get("hop_size", 256)
            ),
            "num_mels": spect_params.get("n_mels", spect_params.get("num_mels", 80)),
            "sampling_rate": sr,
            "fmin": spect_params["fmin"],
            "fmax": None if spect_params["fmax"] == "None" else spect_params["fmax"],
            "center": False,
        }

        assert len(self.data) != 0
        while len(self.data) < batch_size:
            self.data += self.data

    def _build_pairs(self, data_path):
        audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")
        data_path = Path(data_path)

        if data_path.suffix.lower() != ".csv":
            raise ValueError(
                "Only CSV input is supported. Provide a CSV with source,target columns or rows."
            )

        pairs = []
        base_dir = data_path.parent
        with open(data_path, "r", newline="") as f:
            peek = f.readline()
            f.seek(0)
            if "source" in peek and "target" in peek:
                reader = csv.DictReader(f)
                rows = [
                    (row.get("source"), row.get("target"), row.get("split"))
                    for row in reader
                ]
            else:
                reader = csv.reader(f)
                rows = [
                    (row[0], row[1], row[2] if len(row) > 2 else None)
                    for row in reader
                    if len(row) >= 2
                ]
        for src_rel, tgt_rel, row_split in rows:
            if src_rel is None or tgt_rel is None:
                continue
            if self.split and row_split and row_split.lower() != self.split:
                continue
            src = (base_dir / src_rel).expanduser()
            tgt = (base_dir / tgt_rel).expanduser()
            if not src.exists() or not tgt.exists():
                print(f"Warning: skipping missing pair {src} / {tgt}")
                continue
            if (
                src.suffix.lower() not in audio_exts
                or tgt.suffix.lower() not in audio_exts
            ):
                print(f"Warning: unsupported audio extension in pair {src} / {tgt}")
                continue
            pairs.append((src, tgt))
        if len(pairs) == 0:
            raise ValueError(
                "CSV provided but no valid (source,target) pairs found for requested split."
            )
        return pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        src_path, tgt_path = self.data[idx]
        try:
            src_speech, src_orig_sr = librosa.load(src_path, sr=self.sr)
            tgt_speech, tgt_orig_sr = librosa.load(tgt_path, sr=self.sr)
        except Exception as e:
            print(f"Failed to load wav file with error {e}")
            return self.__getitem__(random.randint(0, len(self)))
        if (
            len(src_speech) < self.sr * duration_setting["min"]
            or len(src_speech) > self.sr * duration_setting["max"]
            or len(tgt_speech) < self.sr * duration_setting["min"]
            or len(tgt_speech) > self.sr * duration_setting["max"]
        ):
            print(
                f"Audio pair {src_path} / {tgt_path} is too short or too long, skipping"
            )
            return self.__getitem__(random.randint(0, len(self)))
        if src_orig_sr != self.sr:
            src_speech = librosa.resample(src_speech, src_orig_sr, self.sr)
        if tgt_orig_sr != self.sr:
            tgt_speech = librosa.resample(tgt_speech, tgt_orig_sr, self.sr)

        src_wave = torch.from_numpy(src_speech).float().unsqueeze(0)
        tgt_wave = torch.from_numpy(tgt_speech).float().unsqueeze(0)
        src_mel = to_mel_fn(src_wave, self.mel_fn_args).squeeze(0)
        tgt_mel = to_mel_fn(tgt_wave, self.mel_fn_args).squeeze(0)

        return src_wave.squeeze(0), src_mel, tgt_wave.squeeze(0), tgt_mel


def build_ft_dataloader(
    data_path,
    spect_params,
    sr,
    batch_size=1,
    num_workers=0,
    split="train",
    shuffle=True,
):
    dataset = FT_Dataset(data_path, spect_params, sr, batch_size, split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )
    return dataloader


def collate(batch):
    batch_size = len(batch)

    # sort by target mel length
    lengths = [b[3].shape[1] for b in batch]
    batch_indexes = np.argsort(lengths)[::-1]
    batch = [batch[bid] for bid in batch_indexes]

    n_mels = batch[0][1].size(0)
    max_src_mel_len = max([b[1].shape[1] for b in batch])
    max_tgt_mel_len = max([b[3].shape[1] for b in batch])
    max_src_wave_len = max([b[0].size(0) for b in batch])
    max_tgt_wave_len = max([b[2].size(0) for b in batch])

    src_mels = torch.zeros((batch_size, n_mels, max_src_mel_len)).float() - 10
    tgt_mels = torch.zeros((batch_size, n_mels, max_tgt_mel_len)).float() - 10
    src_waves = torch.zeros((batch_size, max_src_wave_len)).float()
    tgt_waves = torch.zeros((batch_size, max_tgt_wave_len)).float()

    src_mel_lengths = torch.zeros(batch_size).long()
    tgt_mel_lengths = torch.zeros(batch_size).long()
    src_wave_lengths = torch.zeros(batch_size).long()
    tgt_wave_lengths = torch.zeros(batch_size).long()

    for bid, (src_wave, src_mel, tgt_wave, tgt_mel) in enumerate(batch):
        src_mel_size = src_mel.size(1)
        tgt_mel_size = tgt_mel.size(1)
        src_waves[bid, : src_wave.size(0)] = src_wave
        tgt_waves[bid, : tgt_wave.size(0)] = tgt_wave
        src_mels[bid, :, :src_mel_size] = src_mel
        tgt_mels[bid, :, :tgt_mel_size] = tgt_mel
        src_mel_lengths[bid] = src_mel_size
        tgt_mel_lengths[bid] = tgt_mel_size
        src_wave_lengths[bid] = src_wave.size(0)
        tgt_wave_lengths[bid] = tgt_wave.size(0)

    return (
        src_waves,
        src_mels,
        src_wave_lengths,
        src_mel_lengths,
        tgt_waves,
        tgt_mels,
        tgt_wave_lengths,
        tgt_mel_lengths,
    )


if __name__ == "__main__":
    data_path = "./example/reference"
    sr = 22050
    spect_params = {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 80,
        "fmin": 0,
        "fmax": 8000,
    }
    dataloader = build_ft_dataloader(
        data_path, spect_params, sr, batch_size=2, num_workers=0
    )
    for idx, batch in enumerate(dataloader):
        wave, mel, wave_lengths, mel_lengths = batch
        print(wave.shape, mel.shape)
        if idx == 10:
            break
