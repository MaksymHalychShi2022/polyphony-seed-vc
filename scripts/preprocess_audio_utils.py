from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
from ten_vad import TenVad

SINGMOS_SAMPLE_RATE = 16000
VAD_SAMPLE_RATE = 16000
VAD_HOP_SIZE = 256


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32)
    return librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=int(orig_sr),
        target_sr=int(target_sr),
    ).astype(np.float32)


def prepare_singmos_runtime() -> None:
    # SingMOS depends on torchaudio APIs that are absent in recent versions.
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None
    if not hasattr(torchaudio, "sox_effects"):
        sox_module = types.ModuleType("torchaudio.sox_effects")
        sox_module.apply_effects_tensor = (
            lambda waveform, sample_rate, effects, channels_first=True: (
                waveform,
                sample_rate,
            )
        )
        sys.modules["torchaudio.sox_effects"] = sox_module
        torchaudio.sox_effects = sox_module


@lru_cache(maxsize=1)
def load_singmos_predictor() -> Any:
    prepare_singmos_runtime()
    predictor = torch.hub.load(
        "South-Twilight/SingMOS:v1.1.2", "singmos_pro", trust_repo=True
    )
    predictor.eval()
    return predictor


def compute_singmos_mean_for_audio(
    audio: np.ndarray,
    sr: int,
    chunk_seconds: int = 5,
) -> float:
    wave = _resample_audio(audio, sr, SINGMOS_SAMPLE_RATE)
    if wave.size == 0:
        raise ValueError("audio is empty")

    predictor = load_singmos_predictor()
    chunk_samples = max(1, int(chunk_seconds) * SINGMOS_SAMPLE_RATE)
    scores: list[float] = []
    for start in range(0, wave.shape[0], chunk_samples):
        chunk = np.asarray(wave[start : start + chunk_samples], dtype=np.float32)
        if chunk.size == 0:
            continue
        chunk_t = torch.from_numpy(chunk).unsqueeze(0)
        length = torch.tensor([chunk_t.shape[1]], dtype=torch.long)
        with torch.no_grad():
            score = predictor(chunk_t, length)
        scores.append(float(score.item()))

    if not scores:
        raise ValueError("no audio chunks available for SingMOS")
    return float(np.mean(np.asarray(scores, dtype=np.float32)))


@dataclass(frozen=True)
class VadTrack:
    probabilities: np.ndarray
    sample_rate: int = VAD_SAMPLE_RATE
    hop_size: int = VAD_HOP_SIZE

    @property
    def frame_seconds(self) -> float:
        return float(self.hop_size) / float(self.sample_rate)

    def slice_probabilities(self, start_sec: float, end_sec: float) -> np.ndarray:
        if self.probabilities.size == 0:
            return np.zeros(0, dtype=np.float32)
        frame_seconds = self.frame_seconds
        start_idx = max(0, int(np.floor(float(start_sec) / frame_seconds)))
        end_idx = min(
            self.probabilities.size,
            max(start_idx + 1, int(np.ceil(float(end_sec) / frame_seconds))),
        )
        return self.probabilities[start_idx:end_idx]

    def activity_ratio(
        self, start_sec: float, end_sec: float, threshold: float
    ) -> float:
        values = self.slice_probabilities(start_sec, end_sec)
        if values.size == 0:
            return 0.0
        return float(np.mean(values >= float(threshold)))

    def active_regions(
        self,
        threshold: float,
        min_region_seconds: float,
        merge_gap_seconds: float = 0.0,
        pad_seconds: float = 0.0,
    ) -> list[tuple[float, float]]:
        if self.probabilities.size == 0:
            return []

        active = np.asarray(self.probabilities >= float(threshold), dtype=bool)
        frame_seconds = self.frame_seconds
        merge_gap_frames = max(0, int(round(float(merge_gap_seconds) / frame_seconds)))
        if merge_gap_frames > 0:
            idx = 0
            while idx < active.size:
                if active[idx]:
                    idx += 1
                    continue
                end_idx = idx
                while end_idx < active.size and not active[end_idx]:
                    end_idx += 1
                if (
                    0 < idx
                    and end_idx < active.size
                    and (end_idx - idx) <= merge_gap_frames
                ):
                    active[idx:end_idx] = True
                idx = end_idx

        min_frames = max(1, int(np.ceil(float(min_region_seconds) / frame_seconds)))
        pad_frames = max(0, int(round(float(pad_seconds) / frame_seconds)))

        regions: list[tuple[float, float]] = []
        start_idx: int | None = None
        for idx, is_active in enumerate(active):
            if is_active and start_idx is None:
                start_idx = idx
            elif not is_active and start_idx is not None:
                if idx - start_idx >= min_frames:
                    region_start = max(0, start_idx - pad_frames) * frame_seconds
                    region_end = min(active.size, idx + pad_frames) * frame_seconds
                    regions.append((region_start, region_end))
                start_idx = None

        if start_idx is not None and active.size - start_idx >= min_frames:
            region_start = max(0, start_idx - pad_frames) * frame_seconds
            region_end = active.size * frame_seconds
            regions.append((region_start, region_end))

        return regions


def compute_vad_track(audio: np.ndarray, sr: int) -> VadTrack:
    wave = np.asarray(_resample_audio(audio, sr, VAD_SAMPLE_RATE), dtype=np.float32)
    if wave.size == 0:
        return VadTrack(probabilities=np.zeros(0, dtype=np.float32))

    vad = TenVad(hop_size=VAD_HOP_SIZE)
    wave_int16 = np.clip(wave, -1.0, 1.0)
    wave_int16 = (wave_int16 * 32767.0).astype(np.int16)

    probabilities: list[float] = []
    for start in range(0, wave_int16.shape[0], VAD_HOP_SIZE):
        frame = wave_int16[start : start + VAD_HOP_SIZE]
        if frame.shape[0] < VAD_HOP_SIZE:
            frame = np.pad(frame, (0, VAD_HOP_SIZE - frame.shape[0]), mode="constant")
        prob, _ = vad.process(frame)
        probabilities.append(float(prob))

    return VadTrack(probabilities=np.asarray(probabilities, dtype=np.float32))
