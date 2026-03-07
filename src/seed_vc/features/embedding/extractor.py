from pathlib import Path

import librosa
import torch
import torchaudio.compliance.kaldi as kaldi

from seed_vc.features.base import BaseFeatureExtractor
from seed_vc.modules.campplus.DTDNN import CAMPPlus
from seed_vc.utils.hf_utils import load_custom_model_from_hf


class CampplusEmbeddingExtractor(BaseFeatureExtractor):
    feature_name = "embedding"

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
        self._campplus_model: CAMPPlus | None = None

        if not device or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _load_model(self) -> None:
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(
            torch.load(campplus_ckpt_path, map_location="cpu")
        )
        campplus_model.eval()
        campplus_model.to(self.device)
        self._campplus_model = campplus_model

    def _extract(self, audio_path: Path) -> torch.Tensor:
        if self._campplus_model is None:
            self._load_model()

        wave_np, _ = librosa.load(audio_path, sr=16000, mono=True)
        wave_16k = torch.from_numpy(wave_np).float().unsqueeze(0)

        feat = kaldi.fbank(
            wave_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)

        with torch.no_grad():
            embedding = self._campplus_model(feat.unsqueeze(0).to(self.device))

        return embedding.squeeze(0).detach().cpu().to(torch.float32)
