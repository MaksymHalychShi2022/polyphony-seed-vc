"""Generate synthetic CI fixture files for fast offline training tests.

Run once from the project root, then commit the generated files:

    python scripts/generate_ci_fixtures.py
    git add tests/fixtures/
    git commit -m "feat(ci): add synthetic fixture data for CI training"

No model downloads required — all tensors are synthetic random data with
the correct shapes for seed-vc (uvit_whisper_44k model config).
"""

import math
from pathlib import Path

import torch
import torchaudio

# ---------------------------------------------------------------------------
# Config (must match configs/model/default.yaml + uvit_whisper_44k.yaml)
# ---------------------------------------------------------------------------
SR = 44100
HOP_LENGTH = 512
N_MELS = 128
SEMANTIC_DIM = 768  # whisper-small encoder hidden size
EMBEDDING_DIM = 192  # campplus style dim
DURATION_S = 2  # seconds of audio per fixture

T_MEL = math.ceil(DURATION_S * SR / HOP_LENGTH)  # 173
T_SEM = math.ceil(T_MEL / 2)  # 87

FIXTURES_DIR = Path("tests/fixtures")
AUDIO_DIR = FIXTURES_DIR / "audio"
FEATURES_DIR = FIXTURES_DIR / "features"

AUDIO_FILES = {
    "source": AUDIO_DIR / "source.wav",
    "target": AUDIO_DIR / "target.wav",
}

SEED = 42


def _feature_path(audio_path: Path, feature_name: str) -> Path:
    """Replicate BaseFeatureExtractor.get_feature_path() logic."""
    cwd = Path.cwd().resolve()
    resolved = audio_path.expanduser().resolve()
    relative = resolved.relative_to(cwd)
    relative_feature = relative.with_suffix(f"{relative.suffix}.pt")
    return FEATURES_DIR / feature_name / relative_feature


def make_audio(path: Path, freq: float = 440.0) -> None:
    """Write a 2-second mono sine wave WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = torch.linspace(0, DURATION_S, int(SR * DURATION_S))
    wave = (0.5 * torch.sin(2 * math.pi * freq * t)).unsqueeze(0)  # (1, N)
    torchaudio.save(str(path), wave, SR)


def save_feature(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tensor": tensor}, path)


def make_features(audio_path: Path, rng: torch.Generator) -> None:
    # mel: (N_MELS, T_MEL) — values in typical mel range
    mel = torch.randn(N_MELS, T_MEL, generator=rng) * 2 - 5
    save_feature(_feature_path(audio_path, "mel"), mel)

    # semantic: (T_SEM, SEMANTIC_DIM)
    semantic = torch.randn(T_SEM, SEMANTIC_DIM, generator=rng)
    save_feature(_feature_path(audio_path, "semantic"), semantic)

    # f0: (T_MEL,) — Hz values, clamped to [0, 800]
    f0 = torch.rand(T_MEL, generator=rng) * 400 + 100  # 100–500 Hz
    f0 = f0.clamp(0.0, 800.0)
    save_feature(_feature_path(audio_path, "f0"), f0)

    # embedding: (EMBEDDING_DIM,) — L2-normalised
    emb = torch.randn(EMBEDDING_DIM, generator=rng)
    emb = emb / emb.norm()
    save_feature(_feature_path(audio_path, "embedding"), emb)


def make_csvs() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    row = "audio/source.wav,audio/target.wav\n"
    for split in ("train", "val"):
        csv_path = FIXTURES_DIR / f"{split}.csv"
        csv_path.write_text("source,target\n" + row)
        print(f"  wrote {csv_path}")


def main() -> None:
    print("Generating CI fixtures (seed={})...".format(SEED))
    torch.manual_seed(SEED)
    rng = torch.Generator()
    rng.manual_seed(SEED)

    # Audio files
    freqs = {"source": 440.0, "target": 523.25}
    for name, path in AUDIO_FILES.items():
        make_audio(path, freq=freqs[name])
        print(f"  wrote {path}")

    # Feature .pt files
    for path in AUDIO_FILES.values():
        make_features(path, rng)
        for feat in ("mel", "semantic", "f0", "embedding"):
            fp = _feature_path(path, feat)
            print(f"  wrote {fp}  shape={torch.load(fp)['tensor'].shape}")

    # CSVs
    make_csvs()

    print(
        f"\nDone. T_MEL={T_MEL}, T_SEM={T_SEM}, N_MELS={N_MELS}, "
        f"SEMANTIC_DIM={SEMANTIC_DIM}, EMBEDDING_DIM={EMBEDDING_DIM}"
    )
    print("Now run:  git add tests/fixtures/")


if __name__ == "__main__":
    main()
