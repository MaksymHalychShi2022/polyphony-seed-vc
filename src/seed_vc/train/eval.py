import argparse
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
import yaml
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm

from seed_vc.train.ft_dataset import FT_Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated audio against targets using Resemblyzer."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset.csv",
        help="CSV with source,target columns.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        help="Training config used to derive sampling rate and spectrogram params.",
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        default=None,
        help="Directory to store generated audio files. Defaults to a hidden eval cache.",
    )
    parser.add_argument(
        "--generated-suffix",
        type=str,
        default=".wav",
        help="File suffix/extension for generated audio when no template is provided.",
    )
    parser.add_argument(
        "--match-field",
        choices=["source", "target"],
        default="source",
        help="Use source or target filename stem when constructing generated paths.",
    )
    parser.add_argument(
        "--generated-template",
        type=str,
        default=None,
        help=(
            "Optional template for generated filenames. "
            "Available placeholders: {source_stem}, {target_stem}, {source_name}, {target_name}."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for evaluation/inference. Auto-selects CUDA if available.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error when a generated file is missing instead of skipping it.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for generating audio before evaluation.",
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=10, help="Diffusion steps for inference."
    )
    parser.add_argument(
        "--length-adjust",
        type=float,
        default=1.0,
        help="Scaling factor for generated length during inference.",
    )
    parser.add_argument(
        "--cfg-rate",
        type=float,
        default=0.7,
        help="Classifier-free guidance rate for inference.",
    )
    parser.add_argument(
        "--pitch-shift",
        type=int,
        default=0,
        help="Pitch shift (semitones) applied during inference.",
    )
    parser.add_argument(
        "--no-auto-f0-adjust",
        action="store_true",
        help="Disable automatic F0 adjustment during inference.",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation and only evaluate existing generated_dir contents.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path, config_path: Path) -> FT_Dataset:
    config = yaml.safe_load(open(config_path, "r"))
    preprocess_params = config["preprocess_params"]
    sr = preprocess_params.get("sr", 22050)
    spect_params = preprocess_params["spect_params"]
    return FT_Dataset(dataset_path, spect_params, sr=sr, batch_size=1)


def resolve_generated_path(
    src_path: Path, tgt_path: Path, base_dir: Path, args: argparse.Namespace
) -> Path:
    if args.generated_template:
        relative = args.generated_template.format(
            source_stem=src_path.stem,
            target_stem=tgt_path.stem,
            source_name=src_path.name,
            target_name=tgt_path.name,
        )
        return base_dir / relative
    chosen = src_path if args.match_field == "source" else tgt_path
    return base_dir / f"{chosen.stem}{args.generated_suffix}"


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    return float(np.dot(vec_a, vec_b) / denom)


def build_default_generated_dir(dataset_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return dataset_path.parent / ".eval_cache" / timestamp


def setup_inference(args: argparse.Namespace, device: str):
    # Lazy import to keep demo dependencies optional until needed
    from demo import app as demo_app

    demo_app.device = torch.device(device)
    # mimic demo.app CLI args expected by load_models
    model_args = SimpleNamespace(
        fp16=False, checkpoint=args.checkpoint, config=args.config
    )
    (
        demo_app.model_f0,
        demo_app.semantic_fn,
        demo_app.vocoder_fn,
        demo_app.campplus_model,
        demo_app.to_mel_f0,
        demo_app.mel_fn_args,
        demo_app.f0_fn,
    ) = demo_app.load_models(model_args)
    demo_app.max_context_window = demo_app.sr // demo_app.hop_length * 30
    demo_app.overlap_wave_len = demo_app.overlap_frame_len * demo_app.hop_length
    return demo_app


def generate_audio(
    demo_app,
    src_path: Path,
    tgt_path: Path,
    out_path: Path,
    args: argparse.Namespace,
) -> bool:
    try:
        sr, audio = demo_app.voice_conversion(
            str(src_path),
            str(tgt_path),
            args.diffusion_steps,
            args.length_adjust,
            args.cfg_rate,
            not args.no_auto_f0_adjust,
            args.pitch_shift,
        )
        if audio.size == 0:
            raise RuntimeError("Empty audio generated")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        torchaudio.save(out_path, waveform, sr)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to generate for {src_path.name} -> {tgt_path.name}: {exc}")
        return False


def evaluate(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).expanduser()
    config_path = Path(args.config).expanduser()
    generated_base = (
        Path(args.generated_dir).expanduser()
        if args.generated_dir
        else build_default_generated_dir(dataset_path)
    )
    if not generated_base.is_absolute():
        generated_base = dataset_path.parent / generated_base

    try:
        dataset = load_dataset(dataset_path, config_path)
    except ValueError as exc:
        print(f"Failed to load dataset: {exc}")
        return 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = VoiceEncoder(device=device)
    demo_app = None
    if not args.no_generate:
        if not args.checkpoint:
            print("Generation requested but no checkpoint provided.")
            return 1
        demo_app = setup_inference(args, device)

    scores = []
    missing = []
    failures = []

    for src_path, tgt_path in tqdm(dataset.data, desc="Evaluating", unit="pair"):
        gen_path = resolve_generated_path(src_path, tgt_path, generated_base, args)
        if not args.no_generate and demo_app is not None:
            if not generate_audio(demo_app, src_path, tgt_path, gen_path, args):
                failures.append((gen_path, "generation failed"))
                continue
        if not gen_path.exists():
            msg = f"Missing generated audio: {gen_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            missing.append(gen_path)
            continue
        try:
            tgt_wav = preprocess_wav(tgt_path)
            gen_wav = preprocess_wav(gen_path)
            tgt_emb = encoder.embed_utterance(tgt_wav)
            gen_emb = encoder.embed_utterance(gen_wav)
            scores.append(cosine_similarity(tgt_emb, gen_emb))
        except Exception as exc:  # noqa: BLE001
            failures.append((gen_path, str(exc)))
            continue

    if len(scores) == 0:
        print("No valid samples were evaluated.")
        if missing:
            print(f"Missing generated files: {len(missing)}")
        if failures:
            print(f"Failed to process: {len(failures)}")
        return 1

    scores_arr = np.asarray(scores)
    print(
        f"Evaluated {len(scores)} pairs "
        f"(skipped {len(missing)} missing, {len(failures)} failed)."
    )
    print(f"Generated audio cached at: {generated_base}")
    print(
        "Resemblyzer cosine similarity -> "
        f"mean: {scores_arr.mean():.4f}, "
        f"median: {np.median(scores_arr):.4f}, "
        f"std: {scores_arr.std():.4f}, "
        f"min: {scores_arr.min():.4f}, "
        f"max: {scores_arr.max():.4f}"
    )
    if missing:
        sample = [str(p) for p in missing[:5]]
        print(f"Missing generated files (showing up to 5): {sample}")
    if failures:
        sample = [f"{path}: {err}" for path, err in failures[:5]]
        print(f"Failed samples (showing up to 5): {sample}")
    return 0


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(evaluate(cli_args))
