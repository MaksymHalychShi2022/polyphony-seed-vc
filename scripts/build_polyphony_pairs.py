"""
Build (source, target) chunk pairs from the Polyphony Project Hugging Face dataset.

For each multitrack song:
  - Download every mic track from the HF dataset repo (cached by huggingface_hub).
  - Resample to a target sample rate and mix all tracks to a polyphonic WAV.
  - Slice the mix and every source track into aligned chunks (1–30 seconds).
  - Emit a CSV with source,target pairs (chunk-level) for Seed-VC's FT_Dataset.

Example:
    HF_TOKEN=... python scripts/build_polyphony_pairs.py \\
        --repo-id ai-department-lpnu/polyphony-project \\
        --base-prefix data/ \\
        --splits train,test \\
        --out-dir data/seed_vc_pairs \\
        --sr 22050 --chunk-seconds 10
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
DEFAULT_REPO_ID = "ai-department-lpnu/polyphony-project"
MIN_CHUNK_SECONDS = 1.0
MAX_CHUNK_SECONDS = 30.0


def list_song_tracks(
    repo_id: str,
    base_prefix: str,
    token: str | None,
    exts: set[str],
    splits: list[str],
) -> list[tuple[str | None, str, list[str]]]:
    """
    Return list of (split, song_id, [paths]) filtered by prefix/ext.
    Handles both legacy layout data/tracks/<song_id>/... and split layout
    data/<split>/<song_id>/...
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    prefix = base_prefix if base_prefix.endswith("/") else f"{base_prefix}/"
    prefix = prefix.rstrip("/") + "/"
    base_default_split = prefix.strip("/").split("/")[-1].lower()
    if base_default_split not in {"train", "test"}:
        base_default_split = None

    normalized_splits = {s.lower() for s in splits}
    allowed_split = None if "all" in normalized_splits else normalized_splits
    songs: dict[str | None, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in files:
        if not path.startswith(prefix):
            continue
        if Path(path).suffix.lower() not in exts:
            continue
        rest = path[len(prefix) :]
        parts = rest.split("/")
        if len(parts) < 2:
            continue  # not a track file
        first = parts[0].strip()
        current_split: str | None = None

        if first.lower() in {"train", "test"}:
            current_split = first.lower()
            if len(parts) < 3:
                continue
            song_id = parts[1].strip()
        elif first == "tracks":
            if len(parts) < 3:
                continue
            song_id = parts[1].strip()
        else:
            song_id = first  # base prefix already pointed into a split or song

        if not song_id:
            continue

        if current_split is None:
            current_split = base_default_split

        if allowed_split and current_split and current_split not in allowed_split:
            continue

        songs[current_split][song_id].append(path)

    song_items: list[tuple[str | None, str, list[str]]] = []
    for split_key, song_map in songs.items():
        for sid, paths in song_map.items():
            song_items.append((split_key, sid, sorted(paths)))

    song_items.sort(key=lambda item: ((item[0] or ""), item[1]))
    return song_items


def load_audio(path: Path, sr: int) -> np.ndarray:
    """Load audio as mono float32 at the desired sample rate."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="FLOAT")


def mix_tracks(waves: Iterable[np.ndarray]) -> np.ndarray:
    waves = list(waves)
    if not waves:
        raise ValueError("No tracks to mix")
    if len(waves) == 1:
        return waves[0].copy()

    max_len = max(len(w) for w in waves)
    padded = [
        np.pad(w, (0, max_len - len(w)), mode="constant") if len(w) < max_len else w
        for w in waves
    ]
    stacked = np.vstack(padded)
    mix = stacked.sum(axis=0) / max(len(waves), 1)

    # Prevent clipping if any large peaks remain.
    peak = np.max(np.abs(mix))
    if peak > 0.99:
        mix = mix / peak
    return mix.astype(np.float32)


def download_track(repo_id: str, token: str | None, path_in_repo: str) -> Path:
    local_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=path_in_repo, token=token
    )
    return Path(local_path)


def apply_fade(audio: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    """Apply linear fade in/out to avoid edge clicks/spikes."""
    if fade_ms <= 0:
        return audio
    fade_len = int(sr * fade_ms / 1000.0)
    if fade_len <= 0 or fade_len * 2 > len(audio):
        return audio
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    audio[:fade_len] *= fade
    audio[-fade_len:] *= fade[::-1]
    return audio


def normalize_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    """Scale audio so the absolute peak matches target_peak (0..1)."""
    if target_peak <= 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def postprocess_audio(audio: np.ndarray, sr: int, target_peak: float, fade_ms: float):
    audio = normalize_peak(audio, target_peak)
    audio = apply_fade(audio, sr, fade_ms)
    return audio.astype(np.float32)


def build_chunk_ranges(
    total_len: int, sr: int, chunk_seconds: float, min_chunk_seconds: float
) -> list[tuple[int, int]]:
    chunk_len = int(chunk_seconds * sr)
    min_len = int(min_chunk_seconds * sr)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_len:
        end = min(start + chunk_len, total_len)
        if end - start >= min_len:
            ranges.append((start, end))
        start += chunk_len
    return ranges


def is_silent_chunk(
    audio: np.ndarray,
    threshold_db: float,
    frame_length: int = 2048,
    hop_length: int = 512,
    min_voiced_ratio: float = 0.1,
) -> bool:
    if audio.size == 0:
        return True
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    ).squeeze()
    if rms.size == 0:
        return True
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    voiced_ratio = float(np.mean(rms_db > threshold_db))
    return voiced_ratio < min_voiced_ratio


def process_song(
    song_id: str,
    repo_paths: list[str],
    split: str | None,
    args: argparse.Namespace,
    token: str | None,
) -> list[tuple[Path, Path, str | None, str, int, int]]:
    """
    Download/mix one song.
    Returns list of (source_rel, target_rel, split, song_id, num_tracks, chunk_idx).
    """
    out_dir: Path = args.out_dir
    song_dir = out_dir / song_id if not split else out_dir / split / song_id

    if args.skip_existing and song_dir.exists():
        csv_pairs = []
        mix_chunks = sorted(song_dir.glob("mixture_chunk*.wav"))
        if mix_chunks:
            track_chunks = sorted(song_dir.glob("*_chunk*.wav"))
            for src in track_chunks:
                if "mixture" in src.name:
                    continue
                # assume matching mixture chunk exists
                chunk_idx = int(src.stem.split("chunk")[-1])
                tgt = song_dir / f"mixture_chunk{chunk_idx:04d}.wav"
                if tgt.exists():
                    csv_pairs.append(
                        (
                            src.relative_to(out_dir),
                            tgt.relative_to(out_dir),
                            split,
                            song_id,
                            len(repo_paths),
                            chunk_idx,
                        )
                    )
        if csv_pairs:
            return csv_pairs

    local_paths = [download_track(args.repo_id, token, rp) for rp in repo_paths]
    waves = [load_audio(lp, args.sr) for lp in local_paths]
    track_stems = [Path(rp).stem for rp in repo_paths]

    mix_wave = mix_tracks(waves)
    chunk_ranges = build_chunk_ranges(
        len(mix_wave), args.sr, args.chunk_seconds, args.min_chunk_seconds
    )

    if not chunk_ranges:
        return []

    # Save mixture chunks once; reused by all sources.
    mixture_paths: dict[int, Path] = {}
    for chunk_idx, (start, end) in enumerate(chunk_ranges):
        mix_chunk = mix_wave[start:end]
        mix_chunk = postprocess_audio(
            mix_chunk, args.sr, target_peak=args.target_peak, fade_ms=args.fade_ms
        )
        mix_path = song_dir / f"mixture_chunk{chunk_idx:04d}.wav"
        save_audio(mix_path, mix_chunk, args.sr)
        mixture_paths[chunk_idx] = mix_path

    pairs: list[tuple[Path, Path, str, int, int]] = []
    for track_stem, wave in zip(track_stems, waves):
        for chunk_idx, (start, end) in enumerate(chunk_ranges):
            target_len = end - start
            src_chunk = wave[start:end]
            if len(src_chunk) < target_len:
                src_chunk = np.pad(
                    src_chunk, (0, target_len - len(src_chunk)), mode="constant"
                )
            if is_silent_chunk(
                src_chunk,
                threshold_db=args.silence_db,
                frame_length=args.rms_frame_length,
                hop_length=args.rms_hop_length,
                min_voiced_ratio=args.min_voiced_ratio,
            ):
                continue
            src_chunk = postprocess_audio(
                src_chunk, args.sr, target_peak=args.target_peak, fade_ms=args.fade_ms
            )
            src_path = song_dir / f"{track_stem}_chunk{chunk_idx:04d}.wav"
            save_audio(src_path, src_chunk, args.sr)
            pairs.append(
                (
                    src_path.relative_to(out_dir),
                    mixture_paths[chunk_idx].relative_to(out_dir),
                    split,
                    song_id,
                    len(repo_paths),
                    chunk_idx,
                )
            )

    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create (source,target) chunk pairs for Seed-VC finetuning"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token if the dataset is private (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--base-prefix",
        default="data/",
        help='Path prefix inside the HF dataset that holds song folders (e.g. "data/" or "data/train/").',
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/polyphony-project-dataset"),
        help="Where to store chunked WAVs and the pairs.csv file.",
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help='Comma-separated list of splits to process (e.g. "train,test" or "all").',
    )
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate.")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.0,
        help="Chunk duration (must be between 1 and 30 seconds).",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=MIN_CHUNK_SECONDS,
        help="Drop trailing chunks shorter than this (min 1s).",
    )
    parser.add_argument(
        "--min-tracks",
        type=int,
        default=2,
        help="Skip songs with fewer than this many tracks.",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=0,
        help="Process at most this many songs (0 = all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing chunks if they already exist for a song.",
    )
    parser.add_argument(
        "--csv-name",
        default="pairs.csv",
        help="Name of the CSV to write inside out-dir.",
    )
    parser.add_argument(
        "--silence-db",
        type=float,
        default=-45.0,
        help="RMS dB threshold below which a frame is treated as silence.",
    )
    parser.add_argument(
        "--min-voiced-ratio",
        type=float,
        default=0.9,
        help="Require at least this fraction of frames above silence_db to keep a chunk.",
    )
    parser.add_argument(
        "--rms-frame-length",
        type=int,
        default=2048,
        help="Frame length for RMS-based silence detection.",
    )
    parser.add_argument(
        "--rms-hop-length",
        type=int,
        default=512,
        help="Hop length for RMS-based silence detection.",
    )
    parser.add_argument(
        "--target-peak",
        type=float,
        default=0.95,
        help="Peak-normalize each chunk to this linear amplitude (0..1).",
    )
    parser.add_argument(
        "--fade-ms",
        type=float,
        default=8.0,
        help="Apply fade in/out of this length (ms) to each chunk to prevent spikes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or None
    args.out_dir = args.out_dir.resolve()
    args.splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not args.splits:
        args.splits = ["train", "test"]

    if not (MIN_CHUNK_SECONDS <= args.chunk_seconds <= MAX_CHUNK_SECONDS):
        raise SystemExit("chunk-seconds must be between 1 and 30 seconds.")
    if args.min_chunk_seconds < MIN_CHUNK_SECONDS:
        raise SystemExit("min-chunk-seconds must be at least 1 second.")
    if args.min_chunk_seconds > args.chunk_seconds:
        raise SystemExit("min-chunk-seconds cannot exceed chunk-seconds.")

    songs = list_song_tracks(
        args.repo_id, args.base_prefix, token, AUDIO_EXTS, args.splits
    )
    if not songs:
        raise SystemExit("No tracks found; check repo id, token, and base prefix.")

    if args.max_songs > 0:
        split_order: list[str | None] = []
        grouped: dict[str | None, list[tuple[str | None, str, list[str]]]] = (
            defaultdict(list)
        )
        for item in songs:
            split = item[0]
            if split not in grouped:
                split_order.append(split)
            grouped[split].append(item)
        song_items: list[tuple[str | None, str, list[str]]] = []
        for split in split_order:
            song_items.extend(grouped[split][: args.max_songs])
    else:
        song_items = songs

    pairs: list[tuple[Path, Path, str | None, str, int, int]] = []
    for idx, (split, song_id, repo_paths) in enumerate(song_items, start=1):
        if len(repo_paths) < args.min_tracks:
            print(
                f"[{idx}/{len(song_items)}] {song_id}: skipped (<{args.min_tracks} tracks)"
            )
            continue
        try:
            song_pairs = process_song(song_id, repo_paths, split, args, token)
        except Exception as exc:  # noqa: BLE001
            label = f"{split}/{song_id}" if split else song_id
            print(f"[{idx}/{len(song_items)}] {label}: failed ({exc})")
            continue
        pairs.extend(song_pairs)
        label = f"{split}/{song_id}" if split else song_id
        print(
            f"[{idx}/{len(song_items)}] {label}: "
            f"{len(repo_paths)} tracks -> {len(song_pairs)} kept chunks"
        )

    if not pairs:
        raise SystemExit("No pairs were created.")

    csv_path = args.out_dir / args.csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["source", "target", "split", "song_id", "num_tracks", "chunk_idx"]
        )
        for src_rel, tgt_rel, split, song_id, n_tracks, chunk_idx in pairs:
            writer.writerow(
                [
                    src_rel.as_posix(),
                    tgt_rel.as_posix(),
                    split or "",
                    song_id,
                    n_tracks,
                    chunk_idx,
                ]
            )

    print(f"Wrote {len(pairs)} pairs to {csv_path}")
    print(f"Example row: source={pairs[0][0]} target={pairs[0][1]}")


if __name__ == "__main__":
    main()
