"""Process a raw multitrack dataset into aligned chunked FLAC stems + mixture.

Input layout:
  RAW_DIR/<song_id>/<stem_name>.mp3

Output layout:
  OUT_DIR/<song_id>/
    <chunk_id>_<stem_name>.flac
    <chunk_id>_mixture.flac
    preprocess_metadata.json

The pipeline scores songs with SingMOS-Pro, derives candidate intervals from TEN VAD,
filters low-value source/target pairs, and writes dataset-level metadata.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import click
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

from preprocess_audio_utils import compute_singmos_mean_for_audio, compute_vad_track


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


def load_mono(path: Path, sr: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = np.mean(audio, axis=0).astype(np.float32)
    return audio


def pad_to_len(waves: list[np.ndarray], length: int) -> list[np.ndarray]:
    out = []
    for wave in waves:
        if len(wave) == length:
            out.append(wave)
        elif len(wave) < length:
            out.append(np.pad(wave, (0, length - len(wave)), mode="constant"))
        else:
            out.append(wave[:length])
    return out


def mix_sum(waves: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    if not waves:
        raise ValueError("No stems to mix")
    total_len = max(len(wave) for wave in waves)
    waves = pad_to_len(waves, total_len)
    stacked = np.vstack([wave[None, :] for wave in waves])
    return stacked.sum(axis=0).astype(np.float32), waves


def compute_lufs_gain(audio: np.ndarray, sr: int, target_lufs: float) -> float:
    if audio.size == 0:
        return 1.0

    meter = pyln.Meter(sr)
    try:
        current = float(meter.integrated_loudness(audio.astype("float64")))
    except Exception:
        return 1.0

    if not math.isfinite(current):
        return 1.0

    gain_db = float(target_lufs) - current
    return float(10 ** (gain_db / 20.0))


def apply_global_gain(waves: list[np.ndarray], gain: float) -> list[np.ndarray]:
    if gain == 1.0:
        return waves
    return [np.asarray(wave * gain, dtype=np.float32) for wave in waves]


def apply_peak_normalize(waves: list[np.ndarray], peak: float) -> list[np.ndarray]:
    if peak <= 0:
        return waves
    max_peak = 0.0
    for wave in waves:
        if wave.size == 0:
            continue
        max_peak = max(max_peak, float(np.max(np.abs(wave))))
    if max_peak <= 0:
        return waves
    scale = float(peak) / max_peak
    return [np.asarray(wave * scale, dtype=np.float32) for wave in waves]


def silence_ratio_db(
    audio: np.ndarray,
    threshold_db: float,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> float:
    if audio.size == 0:
        return 1.0
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    ).squeeze()
    if rms.size == 0:
        return 1.0
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    return float(np.mean(rms_db < float(threshold_db)))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))


def save_flac(path: Path, audio: np.ndarray, sr: int, subtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, format="FLAC", subtype=subtype)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def split_region(
    start_sec: float,
    end_sec: float,
    preferred_seconds: float,
    min_seconds: float,
    max_seconds: float,
) -> list[tuple[float, float]]:
    total = float(end_sec) - float(start_sec)
    if total < float(min_seconds):
        return []

    preferred = min(
        max(float(preferred_seconds), float(min_seconds)), float(max_seconds)
    )
    segments: list[tuple[float, float]] = []
    cursor = float(start_sec)
    while end_sec - cursor > float(max_seconds):
        segments.append((cursor, cursor + float(max_seconds)))
        cursor += float(max_seconds)

    while end_sec - cursor > preferred + float(min_seconds):
        segments.append((cursor, cursor + preferred))
        cursor += preferred

    remainder = float(end_sec) - cursor
    if remainder >= float(min_seconds):
        segments.append((cursor, float(end_sec)))
    elif segments and float(end_sec) - segments[-1][0] <= float(max_seconds):
        last_start, _ = segments[-1]
        segments[-1] = (last_start, float(end_sec))

    return segments


def derive_candidate_segments(
    audio: np.ndarray,
    sr: int,
    vad_threshold: float,
    min_vad_region_seconds: float,
    vad_merge_gap_seconds: float,
    vad_pad_seconds: float,
    preferred_chunk_seconds: float,
    min_chunk_seconds: float,
    max_chunk_seconds: float,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    vad_track = compute_vad_track(audio, sr)
    regions = vad_track.active_regions(
        threshold=vad_threshold,
        min_region_seconds=min_vad_region_seconds,
        merge_gap_seconds=vad_merge_gap_seconds,
        pad_seconds=vad_pad_seconds,
    )
    segments: list[tuple[float, float]] = []
    for start_sec, end_sec in regions:
        segments.extend(
            split_region(
                start_sec=start_sec,
                end_sec=end_sec,
                preferred_seconds=preferred_chunk_seconds,
                min_seconds=min_chunk_seconds,
                max_seconds=max_chunk_seconds,
            )
        )
    return segments, vad_track.probabilities


def load_song_data(
    song_dir: Path, sr: int
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    stems = list_stem_files(song_dir)
    if not stems:
        raise ValueError("song has no supported stem files")
    stem_names = [path.stem for path in stems]
    waves = [load_mono(path, sr=sr) for path in stems]
    mix, waves = mix_sum(waves)
    return stem_names, waves, mix


def score_song(
    song_dir: Path, sr: int, singmos_chunk_seconds: int
) -> tuple[float, int]:
    stem_names, _, mix = load_song_data(song_dir, sr=sr)
    score = compute_singmos_mean_for_audio(
        mix,
        sr=sr,
        chunk_seconds=singmos_chunk_seconds,
    )
    return score, len(stem_names)


def select_songs(
    song_dirs: list[Path],
    sr: int,
    max_songs: int,
    quality_threshold: float,
    singmos_chunk_seconds: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    scoreable: list[dict[str, Any]] = []

    for song_dir in tqdm(song_dirs, desc="Scoring songs", unit="song"):
        record: dict[str, Any] = {
            "song_name": song_dir.name,
            "selected": False,
        }
        try:
            score, n_stems = score_song(
                song_dir=song_dir,
                sr=sr,
                singmos_chunk_seconds=singmos_chunk_seconds,
            )
            record["quality_score"] = score
            record["input_stem_count"] = n_stems
            scoreable.append(record)
        except Exception as exc:
            record["quality_score"] = None
            record["selection_reason"] = f"score_failed: {exc}"
        records.append(record)

    threshold_enabled = quality_threshold >= 0.0
    eligible = [
        record
        for record in scoreable
        if not threshold_enabled or float(record["quality_score"]) >= quality_threshold
    ]
    eligible.sort(key=lambda item: (-float(item["quality_score"]), item["song_name"]))

    if max_songs > 0:
        selected_names = {record["song_name"] for record in eligible[:max_songs]}
    else:
        selected_names = {record["song_name"] for record in eligible}

    score_ranks = {record["song_name"]: idx + 1 for idx, record in enumerate(eligible)}
    selected_dirs: list[Path] = []
    for song_dir in song_dirs:
        name = song_dir.name
        record = next(item for item in records if item["song_name"] == name)
        if name in selected_names:
            record["selected"] = True
            record["selection_reason"] = "selected"
            record["quality_rank"] = score_ranks[name]
            selected_dirs.append(song_dir)
            continue
        if record.get("quality_score") is None:
            continue
        if threshold_enabled and float(record["quality_score"]) < quality_threshold:
            record["selection_reason"] = "below_quality_threshold"
        else:
            record["selection_reason"] = "rank_exceeds_max_songs"
        record["quality_rank"] = score_ranks.get(name)

    return selected_dirs, records


def process_song(
    song_dir: Path,
    out_dir: Path,
    sr: int,
    preferred_chunk_seconds: float,
    min_chunk_seconds: float,
    max_chunk_seconds: float,
    target_lufs: float,
    peak: float,
    silence_threshold_db: float,
    max_silence_ratio: float,
    subtype: str,
    vad_threshold: float,
    min_vad_region_seconds: float,
    vad_merge_gap_seconds: float,
    vad_pad_seconds: float,
    min_source_activity_ratio: float,
    min_polyphony_ratio: float,
    quality_score: float | None,
) -> tuple[int, int, dict[str, Any]]:
    stem_names, waves, mix = load_song_data(song_dir, sr=sr)
    mix_segments, mix_vad_probabilities = derive_candidate_segments(
        audio=mix,
        sr=sr,
        vad_threshold=vad_threshold,
        min_vad_region_seconds=min_vad_region_seconds,
        vad_merge_gap_seconds=vad_merge_gap_seconds,
        vad_pad_seconds=vad_pad_seconds,
        preferred_chunk_seconds=preferred_chunk_seconds,
        min_chunk_seconds=min_chunk_seconds,
        max_chunk_seconds=max_chunk_seconds,
    )
    stem_vad_tracks = [compute_vad_track(wave, sr=sr) for wave in waves]

    song_out = out_dir / song_dir.name
    song_out.mkdir(parents=True, exist_ok=True)
    kept = 0
    segment_records: list[dict[str, Any]] = []
    for segment_idx, (start_sec, end_sec) in enumerate(mix_segments):
        start = max(0, int(round(start_sec * sr)))
        end = min(len(mix), int(round(end_sec * sr)))
        if end <= start:
            continue

        raw_stem_chunks = [
            np.asarray(stem_wave[start:end], dtype=np.float32) for stem_wave in waves
        ]
        raw_mix_chunk = np.asarray(mix[start:end], dtype=np.float32)
        segment_record: dict[str, Any] = {
            "candidate_index": segment_idx,
            "start_sec": round(start / sr, 4),
            "end_sec": round(end / sr, 4),
            "duration_sec": round((end - start) / sr, 4),
            "source_results": [],
        }

        if silence_ratio_db(raw_mix_chunk, threshold_db=silence_threshold_db) > float(
            max_silence_ratio
        ):
            segment_record["accepted"] = False
            segment_record["rejection_reason"] = "mixture_silence_ratio"
            segment_records.append(segment_record)
            continue

        normalized_stem_chunks = [
            apply_global_gain(
                [stem_chunk],
                compute_lufs_gain(stem_chunk, sr=sr, target_lufs=target_lufs),
            )[0]
            for stem_chunk in raw_stem_chunks
        ]
        normalized_mix_chunk, normalized_stem_chunks = mix_sum(normalized_stem_chunks)
        normalized_chunks = apply_peak_normalize(
            normalized_stem_chunks + [normalized_mix_chunk],
            peak=peak,
        )
        normalized_stem_chunks, normalized_mix_chunk = (
            normalized_chunks[:-1],
            normalized_chunks[-1],
        )

        mix_rms = max(rms(normalized_mix_chunk), 1e-8)
        accepted_sources: list[tuple[str, np.ndarray]] = []
        for stem_name, stem_vad_track, normalized_stem_chunk in zip(
            stem_names,
            stem_vad_tracks,
            normalized_stem_chunks,
        ):
            source_activity_ratio = stem_vad_track.activity_ratio(
                start_sec=start / sr,
                end_sec=end / sr,
                threshold=vad_threshold,
            )
            residual_chunk = normalized_mix_chunk - normalized_stem_chunk
            polyphony_ratio = rms(residual_chunk) / mix_rms
            accepted = True
            rejection_reasons: list[str] = []
            if source_activity_ratio < float(min_source_activity_ratio):
                accepted = False
                rejection_reasons.append("insufficient_source_activity")
            if polyphony_ratio < float(min_polyphony_ratio):
                accepted = False
                rejection_reasons.append("insufficient_polyphonic_contrast")

            source_record = {
                "stem_name": stem_name,
                "accepted": accepted,
                "source_activity_ratio": round(source_activity_ratio, 4),
                "polyphony_ratio": round(polyphony_ratio, 4),
                "rejection_reasons": rejection_reasons,
            }
            segment_record["source_results"].append(source_record)
            if accepted:
                accepted_sources.append((stem_name, normalized_stem_chunk))

        if not accepted_sources:
            segment_record["accepted"] = False
            segment_record["rejection_reason"] = "no_sources_passed_filters"
            segment_records.append(segment_record)
            continue

        chunk_id = f"{kept:06d}"
        for stem_name, normalized_stem_chunk in accepted_sources:
            save_flac(
                song_out / f"{chunk_id}_{stem_name}.flac",
                normalized_stem_chunk,
                sr,
                subtype,
            )
        save_flac(
            song_out / f"{chunk_id}_mixture.flac", normalized_mix_chunk, sr, subtype
        )

        segment_record["accepted"] = True
        segment_record["chunk_id"] = chunk_id
        segment_record["accepted_source_count"] = len(accepted_sources)
        segment_records.append(segment_record)
        kept += 1

    song_metadata = {
        "song_name": song_dir.name,
        "quality_score": quality_score,
        "input_stem_count": len(stem_names),
        "candidate_segment_count": len(mix_segments),
        "written_chunk_count": kept,
        "mix_vad_frame_count": int(mix_vad_probabilities.size),
        "segments": segment_records,
    }
    write_json(song_out / "preprocess_metadata.json", song_metadata)
    return len(stem_names), kept, song_metadata


@click.command(context_settings={"show_default": True})
@click.option("--sr", type=int, default=44100, help="Target sample rate")
@click.option(
    "--chunk-seconds",
    type=float,
    default=8.0,
    help="Preferred chunk duration when splitting long VAD regions",
)
@click.option(
    "--min-chunk-seconds",
    type=float,
    default=1.0,
    help="Minimum accepted chunk duration",
)
@click.option(
    "--max-chunk-seconds",
    type=float,
    default=30.0,
    help="Maximum accepted chunk duration",
)
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
    help="RMS (dB) threshold for mixture silence backstop filtering",
)
@click.option(
    "--max-silence-ratio",
    type=float,
    default=0.05,
    help="Skip chunk if mixture silent-frame fraction exceeds this",
)
@click.option(
    "--max-songs",
    type=int,
    default=-1,
    help="Keep only the top-N songs after quality scoring (-1 = keep all eligible)",
)
@click.option(
    "--quality-threshold",
    type=float,
    default=-1.0,
    help="Minimum SingMOS song score to keep (-1 disables threshold)",
)
@click.option(
    "--singmos-chunk-seconds",
    type=int,
    default=5,
    help="Chunk size used when averaging SingMOS song scores",
)
@click.option(
    "--vad-threshold",
    type=float,
    default=0.5,
    help="TEN VAD probability threshold used for active speech regions",
)
@click.option(
    "--min-vad-region-seconds",
    type=float,
    default=0.5,
    help="Minimum duration of an active VAD region before chunk splitting",
)
@click.option(
    "--vad-merge-gap-seconds",
    type=float,
    default=0.2,
    help="Merge inactive VAD gaps shorter than this duration",
)
@click.option(
    "--vad-pad-seconds",
    type=float,
    default=0.1,
    help="Pad each VAD-derived region on both sides by this duration",
)
@click.option(
    "--min-source-activity-ratio",
    type=float,
    default=0.2,
    help="Minimum fraction of active VAD frames required for a source chunk",
)
@click.option(
    "--min-polyphony-ratio",
    type=float,
    default=0.15,
    help="Minimum residual-energy ratio required for target-vs-source contrast",
)
@click.option(
    "--subtype",
    type=str,
    default="PCM_24",
    help="SoundFile FLAC subtype (e.g. PCM_16, PCM_24)",
)
def main(
    sr: int,
    chunk_seconds: float,
    min_chunk_seconds: float,
    max_chunk_seconds: float,
    target_lufs: float,
    peak: float,
    silence_threshold_db: float,
    max_silence_ratio: float,
    max_songs: int,
    quality_threshold: float,
    singmos_chunk_seconds: int,
    vad_threshold: float,
    min_vad_region_seconds: float,
    vad_merge_gap_seconds: float,
    vad_pad_seconds: float,
    min_source_activity_ratio: float,
    min_polyphony_ratio: float,
    subtype: str,
) -> None:
    if min_chunk_seconds <= 0:
        raise click.ClickException("--min-chunk-seconds must be > 0")
    if max_chunk_seconds < min_chunk_seconds:
        raise click.ClickException("--max-chunk-seconds must be >= --min-chunk-seconds")

    raw_dir = Path(os.environ["DATA_RAW"]).expanduser()
    out_dir = Path(os.environ["DATA_PROCESSED"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    song_dirs = list_song_dirs(raw_dir)
    if not song_dirs:
        raise click.ClickException(f"No song folders found in {raw_dir}")

    selected_song_dirs, selection_records = select_songs(
        song_dirs=song_dirs,
        sr=int(sr),
        max_songs=int(max_songs),
        quality_threshold=float(quality_threshold),
        singmos_chunk_seconds=int(singmos_chunk_seconds),
    )
    if not selected_song_dirs:
        raise click.ClickException("No songs selected after quality filtering")

    record_map = {record["song_name"]: record for record in selection_records}
    processed = 0
    total_chunks = 0
    failed = 0
    it = tqdm(selected_song_dirs, desc="Processing songs", unit="song")
    for i, song_dir in enumerate(it, start=1):
        try:
            n_stems, kept, song_metadata = process_song(
                song_dir=song_dir,
                out_dir=out_dir,
                sr=int(sr),
                preferred_chunk_seconds=float(chunk_seconds),
                min_chunk_seconds=float(min_chunk_seconds),
                max_chunk_seconds=float(max_chunk_seconds),
                target_lufs=float(target_lufs),
                peak=float(peak),
                silence_threshold_db=float(silence_threshold_db),
                max_silence_ratio=float(max_silence_ratio),
                subtype=str(subtype),
                vad_threshold=float(vad_threshold),
                min_vad_region_seconds=float(min_vad_region_seconds),
                vad_merge_gap_seconds=float(vad_merge_gap_seconds),
                vad_pad_seconds=float(vad_pad_seconds),
                min_source_activity_ratio=float(min_source_activity_ratio),
                min_polyphony_ratio=float(min_polyphony_ratio),
                quality_score=record_map[song_dir.name].get("quality_score"),
            )
        except Exception as exc:
            failed += 1
            record_map[song_dir.name]["processing_error"] = str(exc)
            tqdm.write(
                f"[{i}/{len(selected_song_dirs)}] {song_dir.name}: FAILED ({exc})"
            )
            continue

        processed += 1
        total_chunks += kept
        record_map[song_dir.name]["processed"] = True
        record_map[song_dir.name]["written_chunk_count"] = kept
        record_map[song_dir.name]["metadata_path"] = str(
            (out_dir / song_dir.name / "preprocess_metadata.json").resolve()
        )
        record_map[song_dir.name]["candidate_segment_count"] = song_metadata[
            "candidate_segment_count"
        ]
        it.set_postfix({"stems": n_stems, "kept": kept, "failed": failed})

    manifest = {
        "config": {
            "sr": sr,
            "preferred_chunk_seconds": chunk_seconds,
            "min_chunk_seconds": min_chunk_seconds,
            "max_chunk_seconds": max_chunk_seconds,
            "target_lufs": target_lufs,
            "peak": peak,
            "silence_threshold_db": silence_threshold_db,
            "max_silence_ratio": max_silence_ratio,
            "max_songs": max_songs,
            "quality_threshold": quality_threshold,
            "singmos_chunk_seconds": singmos_chunk_seconds,
            "vad_threshold": vad_threshold,
            "min_vad_region_seconds": min_vad_region_seconds,
            "vad_merge_gap_seconds": vad_merge_gap_seconds,
            "vad_pad_seconds": vad_pad_seconds,
            "min_source_activity_ratio": min_source_activity_ratio,
            "min_polyphony_ratio": min_polyphony_ratio,
            "subtype": subtype,
        },
        "summary": {
            "total_song_dirs": len(song_dirs),
            "selected_song_count": len(selected_song_dirs),
            "processed_song_count": processed,
            "failed_song_count": failed,
            "written_chunk_count": total_chunks,
        },
        "songs": selection_records,
    }
    write_json(out_dir / "preprocess_manifest.json", manifest)

    click.echo(
        f"Done. Songs selected: {len(selected_song_dirs)}/{len(song_dirs)}. "
        f"Songs processed: {processed}. Failed: {failed}. Chunks written: {total_chunks}."
    )


if __name__ == "__main__":
    main()
