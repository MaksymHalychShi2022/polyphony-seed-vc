from pathlib import Path

import librosa
import torch
from transformers import AutoFeatureExtractor, WhisperModel

from seed_vc.features.base import BaseFeatureExtractor


class WhisperFeatureExtractor(BaseFeatureExtractor):
    feature_name = "semantic"

    def __init__(
        self,
        whisper_model_name: str,
        features_root: str | Path,
        device: str | None = None,
        require_features: bool = False,
    ):
        super().__init__(
            features_root=features_root,
            require_features=require_features,
        )
        self.whisper_model_name = whisper_model_name
        self._whisper_model: WhisperModel | None = None
        self._whisper_feature_extractor: AutoFeatureExtractor | None = None

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _load_model(self) -> None:
        whisper_model = WhisperModel.from_pretrained(self.whisper_model_name).to(
            self.device
        )
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.whisper_model_name
        )
        if hasattr(whisper_model, "decoder"):
            del whisper_model.decoder

        self._whisper_model = whisper_model
        self._whisper_feature_extractor = whisper_feature_extractor

    def _extract(self, audio_path: Path) -> torch.Tensor:
        if not self._whisper_model or not self._whisper_feature_extractor:
            self._load_model()

        wave_np, _ = librosa.load(audio_path, sr=16000, mono=True)
        wave_16k = torch.from_numpy(wave_np).float().unsqueeze(0)

        inputs = self._whisper_feature_extractor(
            [wave_np],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        input_features = self._whisper_model._mask_input_features(
            inputs.input_features,
            attention_mask=inputs.attention_mask,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._whisper_model.encoder(
                input_features.to(self._whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        semantic = outputs.last_hidden_state.to(torch.float32)
        semantic = semantic[:, : wave_16k.size(-1) // 320 + 1]
        return semantic.squeeze(0).detach().cpu()
