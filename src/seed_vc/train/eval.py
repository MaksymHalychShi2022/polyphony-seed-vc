import os
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torchaudio
import yaml
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm

from seed_vc.train.features_dataset import build_features_dataloader
from seed_vc.train.seed_vc_model import SeedVCModel
from seed_vc.utils.hf_utils import load_custom_model_from_hf

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"


def build_default_generated_dir(dataset_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return dataset_path.parent / ".eval_cache" / timestamp


def resolve_generated_path(
    src_path: Path,
    tgt_path: Path,
    base_dir: Path,
    generated_template: str | None,
    match_field: str,
    generated_suffix: str,
) -> Path:
    if generated_template:
        relative = generated_template.format(
            source_stem=src_path.stem,
            target_stem=tgt_path.stem,
            source_name=src_path.name,
            target_name=tgt_path.name,
        )
        return base_dir / relative
    chosen = src_path if match_field == "source" else tgt_path
    return base_dir / f"{chosen.stem}{generated_suffix}"


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    return float(np.dot(vec_a, vec_b) / denom)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_vocoder(config: dict[str, Any], device: torch.device):
    model_params = config["model_params"]
    vocoder_type = model_params["vocoder"]["type"]

    if vocoder_type == "bigvgan":
        from seed_vc.modules.bigvgan import bigvgan

        bigvgan_name = model_params["vocoder"]["name"]
        vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        vocoder.remove_weight_norm()
        return vocoder.eval().to(device)

    if vocoder_type == "hifigan":
        from seed_vc.modules.hifigan.f0_predictor import ConvRNNF0Predictor
        from seed_vc.modules.hifigan.generator import HiFTGenerator

        hift_config = yaml.safe_load(Path("configs/hifigan.yml").read_text())
        hift_gen = HiFTGenerator(
            **hift_config["hift"],
            f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
        )
        hift_path = load_custom_model_from_hf(
            "FunAudioLLM/CosyVoice-300M", "hift.pt", None
        )
        hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
        return hift_gen.eval().to(device)

    raise ValueError(f"Unsupported vocoder type: {vocoder_type}")


def resolve_checkpoint_path(config: dict[str, Any], checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    pretrained_model = config.get("pretrained_model", "")
    if not pretrained_model:
        raise ValueError("No checkpoint provided and config.pretrained_model is empty.")
    ckpt = load_custom_model_from_hf("Plachta/Seed-VC", pretrained_model, None)
    if isinstance(ckpt, tuple):
        return ckpt[0]
    return ckpt


def adjust_f0(
    src_f0: torch.Tensor,
    tgt_f0: torch.Tensor,
    auto_f0_adjust: bool,
    pitch_shift: int,
) -> torch.Tensor:
    adjusted = src_f0.clone()
    voiced_src = src_f0[src_f0 > 1]
    voiced_tgt = tgt_f0[tgt_f0 > 1]

    if auto_f0_adjust and voiced_src.numel() > 0 and voiced_tgt.numel() > 0:
        log_src = torch.log(src_f0 + 1e-5)
        median_src = torch.median(torch.log(voiced_src + 1e-5))
        median_tgt = torch.median(torch.log(voiced_tgt + 1e-5))
        mask = src_f0 > 1
        adjusted[mask] = torch.exp(log_src[mask] - median_src + median_tgt)

    if pitch_shift != 0:
        factor = 2 ** (pitch_shift / 12)
        mask = src_f0 > 1
        adjusted[mask] = adjusted[mask] * factor

    return adjusted


def generate_audio(
    model: SeedVCModel,
    vocoder,
    src_mel: torch.Tensor,
    src_mel_length: torch.Tensor,
    tgt_mel: torch.Tensor,
    tgt_mel_length: torch.Tensor,
    src_semantic: torch.Tensor,
    tgt_semantic: torch.Tensor,
    src_f0: torch.Tensor,
    tgt_f0: torch.Tensor,
    tgt_embedding: torch.Tensor,
    src_path_name: str,
    tgt_path_name: str,
    out_path: Path,
    device: torch.device,
    diffusion_steps: int,
    length_adjust: float,
    cfg_rate: float,
    auto_f0_adjust: bool,
    pitch_shift: int,
    sample_rate: int,
) -> bool:
    try:
        src_mel = src_mel.to(device)
        tgt_mel = tgt_mel.to(device)
        src_semantic = src_semantic.to(device)
        tgt_semantic = tgt_semantic.to(device)
        src_f0 = src_f0.to(device)
        tgt_f0 = tgt_f0.to(device)
        tgt_embedding = tgt_embedding.to(device=device, dtype=torch.float32)

        source_target_len = max(1, int(src_mel_length.item() * length_adjust))
        source_target_lens = torch.tensor(
            [source_target_len], device=device, dtype=torch.long
        )
        prompt_lens = torch.tensor(
            [int(tgt_mel_length.item())], device=device, dtype=torch.long
        )

        adjusted_src_f0 = adjust_f0(src_f0, tgt_f0, auto_f0_adjust, pitch_shift)

        with torch.inference_mode():
            cond, _, _, _, _ = model.length_regulator(
                src_semantic,
                ylens=source_target_lens,
                f0=adjusted_src_f0,
            )
            prompt_cond, _, _, _, _ = model.length_regulator(
                tgt_semantic,
                ylens=prompt_lens,
                f0=tgt_f0,
            )

            cat_condition = torch.cat([prompt_cond, cond], dim=1)
            cat_lens = torch.tensor(
                [cat_condition.size(1)], device=device, dtype=torch.long
            )
            vc_target = model.cfm.inference(
                cat_condition,
                cat_lens,
                tgt_mel,
                tgt_embedding,
                None,
                diffusion_steps,
                inference_cfg_rate=cfg_rate,
            )
            vc_target = vc_target[:, :, tgt_mel.size(-1) :]
            waveform = vocoder(vc_target.float())

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(out_path, waveform.detach().cpu(), sample_rate)
        return True
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to generate for {src_path_name} -> {tgt_path_name}: {exc}")
        return False


@click.command(context_settings={"show_default": True})
@click.option(
    "--dataset",
    "dataset_path",
    default="data/dataset.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV with source,target columns.",
)
@click.option(
    "--config",
    "config_path",
    default="configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Training config used for model and feature settings.",
)
@click.option(
    "--cache-root",
    default=".cache",
    type=click.Path(file_okay=False, path_type=Path),
    help="Cache root used by feature extractors.",
)
@click.option(
    "--require-cache/--allow-compute-missing",
    default=True,
    help="Require cache files, or compute missing FeaturesDataset features on demand.",
)
@click.option(
    "--generated-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to store generated audio files. Defaults to a hidden eval cache.",
)
@click.option(
    "--generated-suffix",
    default=".wav",
    type=str,
    help="File suffix/extension for generated audio when no template is provided.",
)
@click.option(
    "--match-field",
    default="source",
    type=click.Choice(["source", "target"]),
    help="Use source or target filename stem when constructing generated paths.",
)
@click.option(
    "--generated-template",
    default=None,
    type=str,
    help=(
        "Optional template for generated filenames. "
        "Available placeholders: {source_stem}, {target_stem}, {source_name}, {target_name}."
    ),
)
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Device for evaluation/inference.",
)
@click.option(
    "--strict/--no-strict",
    default=False,
    help="Raise an error when a generated file is missing instead of skipping it.",
)
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Checkpoint path for generating audio before evaluation.",
)
@click.option(
    "--diffusion-steps", default=10, type=int, help="Diffusion steps for inference."
)
@click.option(
    "--length-adjust",
    default=1.0,
    type=float,
    help="Scaling factor for generated length during inference.",
)
@click.option(
    "--cfg-rate",
    default=0.7,
    type=float,
    help="Classifier-free guidance rate for inference.",
)
@click.option(
    "--pitch-shift",
    default=0,
    type=int,
    help="Pitch shift (semitones) applied during inference.",
)
@click.option(
    "--auto-f0-adjust/--no-auto-f0-adjust",
    default=True,
    help="Enable automatic F0 adjustment during inference.",
)
@click.option(
    "--generate/--no-generate",
    default=True,
    help="Generate audio before evaluation.",
)
def main(
    dataset_path: Path,
    config_path: Path,
    cache_root: Path,
    require_cache: bool,
    generated_dir: Path | None,
    generated_suffix: str,
    match_field: str,
    generated_template: str | None,
    device: str,
    strict: bool,
    checkpoint: str | None,
    diffusion_steps: int,
    length_adjust: float,
    cfg_rate: float,
    pitch_shift: int,
    auto_f0_adjust: bool,
    generate: bool,
) -> None:
    config = yaml.safe_load(config_path.read_text())
    preprocess_params = config["preprocess_params"]
    sr = int(preprocess_params.get("sr", 22050))
    spect_params = preprocess_params["spect_params"]

    speech_tokenizer = config["model_params"]["speech_tokenizer"]
    speech_tokenizer_type = speech_tokenizer.get("type", "cosyvoice")
    if speech_tokenizer_type != "whisper":
        raise ValueError(
            f"Unsupported speech tokenizer type: {speech_tokenizer_type}. Expected 'whisper'."
        )
    whisper_model_name = speech_tokenizer["name"]

    resolved_device = resolve_device(device)
    torch_device = torch.device(resolved_device)

    dataloader = build_features_dataloader(
        data_path=dataset_path,
        spect_params=spect_params,
        whisper_model_name=whisper_model_name,
        cache_root=cache_root,
        sr=sr,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        require_cache=require_cache,
        semantic_device=resolved_device,
        f0_device=resolved_device,
        embedding_device=resolved_device,
    )

    generated_base = generated_dir or build_default_generated_dir(dataset_path)
    if not generated_base.is_absolute():
        generated_base = dataset_path.parent / generated_base

    encoder = VoiceEncoder(device=resolved_device)

    model = None
    vocoder = None

    if generate:
        checkpoint_path = resolve_checkpoint_path(config, checkpoint)
        model = SeedVCModel(config["model_params"]).to(torch_device)
        model.load_weights(checkpoint_path)
        model.eval()
        model.setup_caches(max_batch_size=1, max_seq_length=8192)
        vocoder = load_vocoder(config, torch_device)
        click.echo(f"Loaded checkpoint: {checkpoint_path}")

    scores: list[float] = []
    missing: list[Path] = []
    failures: list[tuple[Path, str]] = []

    for batch in tqdm(dataloader, desc="Evaluating", unit="pair"):
        (
            src_mels,
            src_mel_lengths,
            tgt_mels,
            tgt_mel_lengths,
            src_semantics,
            _src_semantic_lengths,
            tgt_semantics,
            _tgt_semantic_lengths,
            src_f0s,
            _src_f0_lengths,
            tgt_f0s,
            _tgt_f0_lengths,
            tgt_embeddings,
            src_paths,
            tgt_paths,
        ) = batch

        src_path = Path(src_paths[0])
        tgt_path = Path(tgt_paths[0])
        gen_path = resolve_generated_path(
            src_path=src_path,
            tgt_path=tgt_path,
            base_dir=generated_base,
            generated_template=generated_template,
            match_field=match_field,
            generated_suffix=generated_suffix,
        )

        if generate:
            assert model is not None
            assert vocoder is not None
            ok = generate_audio(
                model=model,
                vocoder=vocoder,
                src_mel=src_mels,
                src_mel_length=src_mel_lengths[0],
                tgt_mel=tgt_mels,
                tgt_mel_length=tgt_mel_lengths[0],
                src_semantic=src_semantics,
                tgt_semantic=tgt_semantics,
                src_f0=src_f0s,
                tgt_f0=tgt_f0s,
                tgt_embedding=tgt_embeddings,
                src_path_name=src_path.name,
                tgt_path_name=tgt_path.name,
                out_path=gen_path,
                device=torch_device,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                cfg_rate=cfg_rate,
                auto_f0_adjust=auto_f0_adjust,
                pitch_shift=pitch_shift,
                sample_rate=sr,
            )
            if not ok:
                failures.append((gen_path, "generation failed"))
                continue

        if not gen_path.exists():
            msg = f"Missing generated audio: {gen_path}"
            if strict:
                raise FileNotFoundError(msg)
            missing.append(gen_path)
            continue

        try:
            tgt_wav = preprocess_wav(tgt_path)
            gen_wav = preprocess_wav(gen_path)
            tgt_emb = np.asarray(
                encoder.embed_utterance(tgt_wav), dtype=np.float32
            ).reshape(-1)
            gen_emb = np.asarray(
                encoder.embed_utterance(gen_wav), dtype=np.float32
            ).reshape(-1)
            scores.append(cosine_similarity(tgt_emb, gen_emb))
        except Exception as exc:  # noqa: BLE001
            failures.append((gen_path, str(exc)))

    if len(scores) == 0:
        click.echo("No valid samples were evaluated.")
        if missing:
            click.echo(f"Missing generated files: {len(missing)}")
        if failures:
            click.echo(f"Failed to process: {len(failures)}")
        raise SystemExit(1)

    scores_arr = np.asarray(scores)
    click.echo(
        f"Evaluated {len(scores)} pairs "
        f"(skipped {len(missing)} missing, {len(failures)} failed)."
    )
    click.echo(f"Generated audio cached at: {generated_base}")
    click.echo(
        "Resemblyzer cosine similarity -> "
        f"mean: {scores_arr.mean():.4f}, "
        f"median: {np.median(scores_arr):.4f}, "
        f"std: {scores_arr.std():.4f}, "
        f"min: {scores_arr.min():.4f}, "
        f"max: {scores_arr.max():.4f}"
    )
    if missing:
        sample = [str(p) for p in missing[:5]]
        click.echo(f"Missing generated files (showing up to 5): {sample}")
    if failures:
        sample = [f"{path}: {err}" for path, err in failures[:5]]
        click.echo(f"Failed samples (showing up to 5): {sample}")


if __name__ == "__main__":
    main()
