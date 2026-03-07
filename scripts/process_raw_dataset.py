"""Process a raw multitrack dataset into aligned chunked FLAC stems + mixture.

Input layout:
  RAW_DIR/<song_id>/<stem_name>.mp3

Output layout:
  OUT_DIR/<song_id>/
    <chunk_id>_<stem_name>.flac
    <chunk_id>_mixture.flac

Chunks are aligned by <chunk_id>. Chunks that contain silence are skipped.
"""

import math
from pathlib import Path

import click
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm


def list_song_dirs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise SystemExit(f"RAW_DIR does not exist or is not a directory: {raw_dir}")
    song_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    song_dirs.sort(key=lambda p: p.name)
    return song_dirs


def list_stem_files(song_dir: Path) -> list[Path]:
    exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
    files = [p for p in song_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def load_mono(path: Path, sr: int):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = np.mean(audio, axis=0).astype(np.float32)
    return audio


def pad_to_len(waves, length: int):
    out = []
    for w in waves:
        if len(w) == length:
            out.append(w)
        elif len(w) < length:
            out.append(np.pad(w, (0, length - len(w)), mode="constant"))
        else:
            out.append(w[:length])
    return out


def mix_sum(waves):
    if not waves:
        raise ValueError("No stems to mix")
    total_len = max(len(w) for w in waves)
    waves = pad_to_len(waves, total_len)
    stacked = np.vstack([w[None, :] for w in waves])
    return stacked.sum(axis=0).astype(np.float32), waves


def compute_lufs_gain(mix, sr: int, target_lufs: float) -> float:
    if mix.size == 0:
        return 1.0

    meter = pyln.Meter(sr)
    try:
        current = float(meter.integrated_loudness(mix.astype("float64")))
    except Exception:
        return 1.0

    if not math.isfinite(current):
        return 1.0

    gain_db = float(target_lufs) - current
    return float(10 ** (gain_db / 20.0))


def apply_global_gain(waves, gain: float):
    if gain == 1.0:
        return waves
    return [np.asarray(w * gain, dtype=np.float32) for w in waves]


def apply_peak_normalize(waves, peak: float):
    if peak <= 0:
        return waves
    max_peak = 0.0
    for w in waves:
        if w.size == 0:
            continue
        max_peak = max(max_peak, float(np.max(np.abs(w))))
    if max_peak <= 0:
        return waves
    scale = float(peak) / max_peak
    return [np.asarray(w * scale, dtype=np.float32) for w in waves]


def silence_ratio_db(
    audio,
    threshold_db: float,
    frame_length: int = 2048,
    hop_length: int = 512,
):
    if audio.size == 0:
        return 1.0
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    ).squeeze()
    if rms.size == 0:
        return 1.0
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    return float(np.mean(rms_db < float(threshold_db)))


def save_flac(path: Path, audio, sr: int, subtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, format="FLAC", subtype=subtype)


def process_song(
    song_dir: Path,
    out_dir: Path,
    sr: int,
    chunk_seconds: float,
    target_lufs: float,
    peak: float,
    silence_threshold_db: float,
    max_silence_ratio: float,
    subtype: str,
) -> tuple[int, int]:
    stems = list_stem_files(song_dir)
    if not stems:
        return 0, 0

    stem_names = [p.stem for p in stems]
    waves = [load_mono(p, sr=sr) for p in stems]
    mix, waves = mix_sum(waves)

    chunk_len = int(round(float(chunk_seconds) * int(sr)))
    if chunk_len <= 0:
        raise ValueError("chunk_seconds must be > 0")

    n_total = len(mix)
    n_windows = n_total // chunk_len
    if n_windows <= 0:
        return 0, 0

    song_out = out_dir / song_dir.name
    kept = 0
    for chunk_idx in range(n_windows):
        start = chunk_idx * chunk_len
        end = start + chunk_len
        mix_chunk = np.asarray(mix[start:end], dtype=np.float32)

        sratio = silence_ratio_db(mix_chunk, threshold_db=silence_threshold_db)
        if sratio > float(max_silence_ratio):
            continue

        stem_chunks = [
            np.asarray(stem_wave[start:end], dtype=np.float32) for stem_wave in waves
        ]

        # Normalize each stem chunk independently to reduce source/target mismatch.
        stem_chunks = [
            apply_global_gain(
                [stem_chunk],
                compute_lufs_gain(stem_chunk, sr=sr, target_lufs=target_lufs),
            )[0]
            for stem_chunk in stem_chunks
        ]

        # Mixture is always the exact sum of normalized stems.
        mix_chunk, stem_chunks = mix_sum(stem_chunks)

        # Final peak normalization uses one shared gain to avoid clipping.
        norm_chunks = apply_peak_normalize(stem_chunks + [mix_chunk], peak=peak)
        stem_chunks, mix_chunk = norm_chunks[:-1], norm_chunks[-1]

        chunk_id = f"{chunk_idx:06d}"
        # Write stems first, then mixture.
        for stem_name, stem_chunk in zip(stem_names, stem_chunks):
            out_path = song_out / f"{chunk_id}_{stem_name}.flac"
            save_flac(out_path, stem_chunk, sr=sr, subtype=subtype)

        mix_path = song_out / f"{chunk_id}_mixture.flac"
        save_flac(mix_path, mix_chunk, sr=sr, subtype=subtype)
        kept += 1

    return len(stems), kept


@click.command(context_settings={"show_default": True})
@click.argument(
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("training_dataset"),
    help="Output directory (created if missing)",
)
@click.option("--sr", type=int, default=44100, help="Target sample rate")
@click.option("--chunk-seconds", type=float, default=8.0, help="Chunk length (s)")
@click.option(
    "--target-lufs",
    type=float,
    default=-14.0,
    help="Per-stem per-chunk target integrated loudness (LUFS)",
)
@click.option(
    "--peak",
    type=float,
    default=0.99,
    help="Per-chunk target peak after stem LUFS normalization (0 disables)",
)
@click.option(
    "--silence-threshold-db",
    type=float,
    default=-30.0,
    help="RMS (dB) threshold for silence detection",
)
@click.option(
    "--max-silence-ratio",
    type=float,
    default=0.05,
    help="Skip chunk if fraction of silent frames exceeds this (0 = any silence)",
)
@click.option(
    "--max-songs",
    type=int,
    default=-1,
    help="Process only first N song ids (-1 = all)",
)
@click.option(
    "--subtype",
    type=str,
    default="PCM_24",
    help="SoundFile FLAC subtype (e.g. PCM_16, PCM_24)",
)
def main(
    raw_dir: Path,
    out_dir: Path,
    sr: int,
    chunk_seconds: float,
    target_lufs: float,
    peak: float,
    silence_threshold_db: float,
    max_silence_ratio: float,
    max_songs: int,
    subtype: str,
) -> None:
    raw_dir = raw_dir.expanduser()
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    song_dirs = list_song_dirs(raw_dir)
    if int(max_songs) > 0:
        song_dirs = song_dirs[: int(max_songs)]

    total_songs = len(song_dirs)
    if total_songs == 0:
        raise click.ClickException(f"No song folders found in {raw_dir}")

    processed = 0
    total_chunks = 0
    failed = 0
    it = tqdm(song_dirs, desc="Processing songs", unit="song")
    for i, song_dir in enumerate(it, start=1):
        try:
            n_stems, kept = process_song(
                song_dir=song_dir,
                out_dir=out_dir,
                sr=int(sr),
                chunk_seconds=float(chunk_seconds),
                target_lufs=float(target_lufs),
                peak=float(peak),
                silence_threshold_db=float(silence_threshold_db),
                max_silence_ratio=float(max_silence_ratio),
                subtype=str(subtype),
            )
        except Exception as e:
            failed += 1
            tqdm.write(f"[{i}/{total_songs}] {song_dir.name}: FAILED ({e})")
            continue

        processed += 1
        total_chunks += kept
        it.set_postfix({"stems": n_stems, "kept": kept, "failed": failed})

    click.echo(
        f"Done. Songs processed: {processed}/{total_songs}. Failed: {failed}. Chunks written: {total_chunks}."
    )


if __name__ == "__main__":
    main()
