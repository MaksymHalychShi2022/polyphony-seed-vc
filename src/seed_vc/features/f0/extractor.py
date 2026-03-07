from pathlib import Path

import librosa
import torch

from seed_vc.features.base import BaseFeatureExtractor
from seed_vc.modules.rmvpe import RMVPE
from seed_vc.utils.hf_utils import load_custom_model_from_hf


class F0FeatureExtractor(BaseFeatureExtractor):
    feature_name = "f0"

    def __init__(
        self,
        cache_root: str | Path,
        device: str | None = None,
        require_cache: bool = False,
    ):
        super().__init__(
            cache_root=cache_root,
            require_cache=require_cache,
        )
        self._rmvpe: RMVPE | None = None

        if not device or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def _load_model(self) -> None:
        model_path = load_custom_model_from_hf(
            "lj1995/VoiceConversionWebUI", "rmvpe.pt", None
        )
        self._rmvpe = RMVPE(model_path, is_half=False, device=self.device)

    def _extract(self, audio_path: Path) -> torch.Tensor:
        if self._rmvpe is None:
            self._load_model()

        wave_np, _ = librosa.load(audio_path, sr=16000, mono=True)
        f0 = self._rmvpe.infer_from_audio(wave_np)
        return torch.from_numpy(f0).to(torch.float32)
