from pathlib import Path
from typing import Any

import librosa
import torch

from seed_vc.features.base import BaseFeatureExtractor
from seed_vc.modules.audio import mel_spectrogram


class MelSpectrogramExtractor(BaseFeatureExtractor):
    feature_name = "mel"

    def __init__(
        self,
        spect_params: dict[str, Any],
        features_root: str | Path,
        sr: int = 22050,
        require_features: bool = False,
    ):
        super().__init__(
            features_root=features_root,
            require_features=require_features,
        )

        self.sr = int(sr)
        self.mel_fn_args = {
            "n_fft": spect_params["n_fft"],
            "win_size": spect_params.get(
                "win_length", spect_params.get("win_size", 1024)
            ),
            "hop_size": spect_params.get(
                "hop_length", spect_params.get("hop_size", 256)
            ),
            "num_mels": spect_params.get("n_mels", spect_params.get("num_mels", 80)),
            "sampling_rate": self.sr,
            "fmin": spect_params["fmin"],
            "fmax": None if spect_params["fmax"] == "None" else spect_params["fmax"],
            "center": False,
        }

    def _extract(self, audio_path: Path) -> torch.Tensor:
        wave_np, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        wave = torch.from_numpy(wave_np).float().unsqueeze(0)
        mel = mel_spectrogram(wave, **self.mel_fn_args).squeeze(0)
        return mel.detach().cpu().to(torch.float32)
