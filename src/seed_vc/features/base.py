from abc import ABC, abstractmethod
from pathlib import Path

import torch

FEATURES_CACHE_SUBDIR = "features"


class BaseFeatureExtractor(ABC):
    feature_name: str = ""

    def __init__(
        self,
        cache_root: str | Path,
        require_cache: bool = False,
    ):
        if not self.feature_name:
            raise ValueError("Extractor must define class attribute 'feature_name'")

        self.cache_root = Path(cache_root).expanduser().resolve()
        self.require_cache = require_cache

    @abstractmethod
    def _extract(self, audio_path: Path) -> torch.Tensor:
        raise NotImplementedError

    def get_cache_path(self, audio_path: str | Path) -> Path:
        resolved = Path(audio_path).expanduser().resolve()
        cwd = Path.cwd().resolve()
        try:
            relative_path = resolved.relative_to(cwd)
        except ValueError as exc:
            raise ValueError(
                f"Audio path must be under project root {cwd}: {resolved}"
            ) from exc

        if relative_path.suffix:
            relative_cache_path = relative_path.with_suffix(
                f"{relative_path.suffix}.pt"
            )
        else:
            relative_cache_path = relative_path.with_suffix(".pt")

        return (
            self.cache_root
            / FEATURES_CACHE_SUBDIR
            / self.feature_name
            / relative_cache_path
        )

    def save_feature(self, cache_path: Path, tensor: torch.Tensor) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tensor": tensor.detach().cpu()}
        torch.save(payload, cache_path)

    def load_feature(self, cache_path: Path) -> torch.Tensor:
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict):
            if "tensor" not in payload:
                raise ValueError(
                    f"Invalid cache payload at {cache_path}: missing 'tensor'"
                )
            return payload["tensor"]
        if torch.is_tensor(payload):
            return payload
        raise ValueError(f"Invalid cache payload at {cache_path}")

    def extract(
        self, audio_path: str | Path, require_cache: bool | None = None
    ) -> torch.Tensor:
        resolved = Path(audio_path).expanduser().resolve()
        cache_path = self.get_cache_path(resolved)

        if cache_path.exists():
            return self.load_feature(cache_path)

        effective_require_cache = (
            self.require_cache if require_cache is None else require_cache
        )
        if effective_require_cache:
            raise FileNotFoundError(
                f"Cached {self.feature_name} feature not found for {resolved}: {cache_path}"
            )

        if not resolved.exists():
            raise FileNotFoundError(f"Audio file not found: {resolved}")

        tensor = self._extract(resolved)
        if not torch.is_tensor(tensor):
            raise TypeError(f"Extractor {self.feature_name} returned non-tensor output")

        self.save_feature(cache_path, tensor)
        return tensor.detach().cpu()
