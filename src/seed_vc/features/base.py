from abc import ABC, abstractmethod
from pathlib import Path

import torch


class BaseFeatureExtractor(ABC):
    feature_name: str = ""

    def __init__(
        self,
        features_root: str | Path,
        require_features: bool = False,
    ):
        if not self.feature_name:
            raise ValueError("Extractor must define class attribute 'feature_name'")

        self.features_root = Path(features_root).expanduser().resolve()
        self.require_features = require_features

    @abstractmethod
    def _extract(self, audio_path: Path) -> torch.Tensor:
        raise NotImplementedError

    def get_feature_path(self, audio_path: str | Path) -> Path:
        resolved = Path(audio_path).expanduser().resolve()
        cwd = Path.cwd().resolve()
        try:
            relative_path = resolved.relative_to(cwd)
        except ValueError as exc:
            raise ValueError(
                f"Audio path must be under project root {cwd}: {resolved}"
            ) from exc

        if relative_path.suffix:
            relative_feature_path = relative_path.with_suffix(
                f"{relative_path.suffix}.pt"
            )
        else:
            relative_feature_path = relative_path.with_suffix(".pt")

        return self.features_root / self.feature_name / relative_feature_path

    def save_feature(self, feature_path: Path, tensor: torch.Tensor) -> None:
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tensor": tensor.detach().cpu()}
        torch.save(payload, feature_path)

    def load_feature(self, feature_path: Path) -> torch.Tensor:
        payload = torch.load(feature_path, map_location="cpu")
        if isinstance(payload, dict):
            if "tensor" not in payload:
                raise ValueError(
                    f"Invalid feature payload at {feature_path}: missing 'tensor'"
                )
            return payload["tensor"]
        if torch.is_tensor(payload):
            return payload
        raise ValueError(f"Invalid feature payload at {feature_path}")

    def extract(
        self, audio_path: str | Path, require_features: bool | None = None
    ) -> torch.Tensor:
        resolved = Path(audio_path).expanduser().resolve()
        feature_path = self.get_feature_path(resolved)

        if feature_path.exists():
            return self.load_feature(feature_path)

        effective_require = (
            self.require_features if require_features is None else require_features
        )
        if effective_require:
            raise FileNotFoundError(
                f"Feature file not found for {resolved}: {feature_path}"
            )

        if not resolved.exists():
            raise FileNotFoundError(f"Audio file not found: {resolved}")

        tensor = self._extract(resolved)
        if not torch.is_tensor(tensor):
            raise TypeError(f"Extractor {self.feature_name} returned non-tensor output")

        self.save_feature(feature_path, tensor)
        return tensor.detach().cpu()
