import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torchaudio
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm

from seed_vc.features.f0.extractor import F0FeatureExtractor
from seed_vc.train.features_dataset import build_features_dataloader
from seed_vc.train.seed_vc_model import SeedVCModel
from seed_vc.utils.hf_utils import load_custom_model_from_hf

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

MANIFEST_SCHEMA_VERSION = 2
DEFAULT_METRIC_KEY = "resemblyzer_similarity"
DEFAULT_RESULTS_MANIFEST_NAME = "results_manifest.json"
DEFAULT_METRICS_MANIFEST_NAME = "metrics_manifest.json"
DEFAULT_REPORT_NAME = "evaluation_report.html"
DEFAULT_LISTENING_GUIDANCE = [
    "Check whether the generated output preserves the source melody contour and phrasing.",
    "Listen for choral timbre similarity against the target mix, not just speaker similarity.",
    "Listen for artifacts such as metallic tone, blurred attacks, unstable pitch, or smeared harmonics.",
    "Use SingMOS as a naturalness hint when available, not as a replacement for listening review.",
]
METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "resemblyzer_similarity": {
        "label": "Resemblyzer similarity",
        "short_label": "Timbre",
        "category": "Timbre similarity",
        "description": "Cosine similarity between target and generated Resemblyzer embeddings.",
        "direction": "higher",
        "decimals": 4,
        "optional": False,
        "enabled_by_default": True,
    },
    "f0_rmse": {
        "label": "F0 RMSE",
        "short_label": "F0 RMSE",
        "category": "Melody preservation",
        "description": "Root-mean-square error between aligned source and generated F0 contours.",
        "direction": "lower",
        "decimals": 3,
        "optional": False,
        "enabled_by_default": True,
    },
    "f0_correlation": {
        "label": "F0 correlation",
        "short_label": "F0 corr",
        "category": "Melody preservation",
        "description": "Pearson correlation between aligned source and generated F0 contours.",
        "direction": "higher",
        "decimals": 4,
        "optional": False,
        "enabled_by_default": True,
    },
    "singmos_naturalness": {
        "label": "SingMOS naturalness",
        "short_label": "SingMOS",
        "category": "Naturalness",
        "description": "Mean SingMOS-Pro score across 5-second generated-audio chunks.",
        "direction": "higher",
        "decimals": 4,
        "optional": True,
        "enabled_by_default": True,
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def relativize_for_report(path_str: str, report_path: Path) -> str:
    target = Path(path_str)
    report_dir = report_path.parent
    try:
        return os.path.relpath(target.resolve(), report_dir.resolve())
    except Exception:  # noqa: BLE001
        return path_str


def resolve_default_artifacts(
    dataset_path: Path,
    generated_dir: Path | None,
    results_manifest: Path | None,
    metrics_manifest: Path | None,
    report_path: Path | None,
) -> tuple[Path, Path, Path, Path]:
    generated_base = generated_dir or build_default_generated_dir(dataset_path)
    if not generated_base.is_absolute():
        generated_base = dataset_path.parent / generated_base

    resolved_results = results_manifest or (
        generated_base / DEFAULT_RESULTS_MANIFEST_NAME
    )
    resolved_metrics = metrics_manifest or (
        generated_base / DEFAULT_METRICS_MANIFEST_NAME
    )
    resolved_report = report_path or (generated_base / DEFAULT_REPORT_NAME)

    return generated_base, resolved_results, resolved_metrics, resolved_report


def build_metric_definitions(enable_singmos: bool) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for key, meta in METRIC_DEFINITIONS.items():
        copied = dict(meta)
        copied["key"] = key
        copied["enabled"] = (
            bool(enable_singmos) if key == "singmos_naturalness" else True
        )
        definitions[key] = copied
    return definitions


def normalize_metric_definitions(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("metric_definitions") or {}
    if not raw:
        legacy = build_metric_definitions(enable_singmos=False)
        if DEFAULT_METRIC_KEY in legacy:
            return {DEFAULT_METRIC_KEY: legacy[DEFAULT_METRIC_KEY]}
        return legacy

    definitions: dict[str, dict[str, Any]] = {}
    for key, meta in raw.items():
        default_meta = dict(METRIC_DEFINITIONS.get(key, {}))
        default_meta.update(meta)
        default_meta["key"] = key
        default_meta.setdefault("label", key)
        default_meta.setdefault("short_label", default_meta["label"])
        default_meta.setdefault("direction", "higher")
        default_meta.setdefault("decimals", 4)
        default_meta.setdefault("enabled", True)
        default_meta.setdefault("optional", False)
        definitions[key] = default_meta
    return definitions


def metric_sort_value(value: float | None, direction: str) -> float | None:
    if value is None:
        return None
    return value if direction == "higher" else -value


def format_metric_value(metric_key: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    meta = METRIC_DEFINITIONS.get(metric_key, {})
    decimals = int(meta.get("decimals", 4))
    return f"{value:.{decimals}f}"


def build_metric_summary(
    metric_key: str,
    values: list[float],
    status_counts: dict[str, int],
    metric_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(METRIC_DEFINITIONS.get(metric_key, {}))
    if metric_meta:
        meta.update(metric_meta)
    summary: dict[str, Any] = {
        "key": metric_key,
        "label": meta.get("label", metric_key),
        "short_label": meta.get("short_label", meta.get("label", metric_key)),
        "category": meta.get("category", "Other"),
        "description": meta.get("description"),
        "direction": meta.get("direction", "higher"),
        "optional": bool(meta.get("optional", False)),
        "enabled": bool(meta.get("enabled", True)),
        "decimals": int(meta.get("decimals", 4)),
        "status_counts": status_counts,
        "scored": len(values),
    }
    if values:
        arr = np.asarray(values, dtype=np.float32)
        summary.update(
            {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    return summary


def ensure_item_metric_maps(item: dict[str, Any]) -> None:
    item.setdefault("metrics", {})
    item.setdefault("metric_statuses", {})
    item.setdefault("metric_errors", {})


def set_metric_result(
    item: dict[str, Any],
    metric_key: str,
    *,
    status: str,
    value: float | None = None,
    error: str | None = None,
) -> None:
    ensure_item_metric_maps(item)
    if value is None:
        item["metrics"].pop(metric_key, None)
    else:
        item["metrics"][metric_key] = value
    item["metric_statuses"][metric_key] = status
    if error:
        item["metric_errors"][metric_key] = error
    else:
        item["metric_errors"].pop(metric_key, None)


def load_audio_mono(audio_path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def align_f0_contours(
    source_f0: np.ndarray, generated_f0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target_len = min(len(source_f0), len(generated_f0))
    if target_len < 2:
        raise ValueError("not enough F0 frames to compare")

    def _resample(values: np.ndarray) -> np.ndarray:
        if len(values) == target_len:
            return values.astype(np.float32, copy=False)
        old_positions = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
        new_positions = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
        return np.interp(new_positions, old_positions, values).astype(np.float32)

    return _resample(source_f0), _resample(generated_f0)


def compute_f0_metrics(
    source_f0: np.ndarray, generated_f0: np.ndarray
) -> dict[str, float]:
    aligned_source, aligned_generated = align_f0_contours(source_f0, generated_f0)
    voiced_mask = (aligned_source > 1.0) & (aligned_generated > 1.0)
    if int(voiced_mask.sum()) < 2:
        raise ValueError("insufficient shared voiced frames for F0 metrics")

    source_voiced = aligned_source[voiced_mask]
    generated_voiced = aligned_generated[voiced_mask]
    rmse = float(np.sqrt(np.mean((source_voiced - generated_voiced) ** 2)))

    if np.std(source_voiced) < 1e-8 or np.std(generated_voiced) < 1e-8:
        raise ValueError("insufficient F0 variance for correlation")
    correlation = float(np.corrcoef(source_voiced, generated_voiced)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("invalid F0 correlation")

    return {"f0_rmse": rmse, "f0_correlation": correlation}


def prepare_singmos_runtime() -> None:
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


def load_singmos_predictor() -> Any:
    prepare_singmos_runtime()
    predictor = torch.hub.load(
        "South-Twilight/SingMOS:v1.1.2", "singmos_pro", trust_repo=True
    )
    predictor.eval()
    return predictor


def compute_singmos_mean(
    predictor: Any,
    audio_path: Path,
    chunk_seconds: int = 5,
    sample_rate: int = 16000,
) -> float:
    waveform, input_sr = load_audio_mono(audio_path)
    if input_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, input_sr, sample_rate)
    waveform = waveform.squeeze(0)
    chunk_samples = chunk_seconds * sample_rate
    scores: list[float] = []
    for start in range(0, waveform.numel(), chunk_samples):
        chunk = waveform[start : start + chunk_samples]
        if chunk.numel() == 0:
            continue
        chunk = chunk.to(torch.float32).unsqueeze(0)
        length = torch.tensor([chunk.shape[1]], dtype=torch.long)
        with torch.no_grad():
            score = predictor(chunk, length)
        scores.append(float(score.item()))
    if not scores:
        raise ValueError("no audio chunks available for SingMOS")
    return float(np.mean(np.asarray(scores, dtype=np.float32)))


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
        if isinstance(hift_path, tuple):
            hift_path = hift_path[0]
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
) -> tuple[bool, str | None]:
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
        return True, None
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to generate for {src_path_name} -> {tgt_path_name}: {exc}")
        return False, str(exc)


def run_generate_results_stage(
    dataset_path: Path,
    split: str,
    config_path: Path,
    require_features: bool,
    generated_base: Path,
    generated_suffix: str,
    match_field: str,
    generated_template: str | None,
    device: str,
    checkpoint: str | None,
    diffusion_steps: int,
    length_adjust: float,
    cfg_rate: float,
    pitch_shift: int,
    auto_f0_adjust: bool,
    should_generate_audio: bool,
    output_manifest_path: Path,
) -> dict[str, Any]:
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
        split=split,
        spect_params=spect_params,
        whisper_model_name=whisper_model_name,
        sr=sr,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        require_features=require_features,
        semantic_device=resolved_device,
        f0_device=resolved_device,
        embedding_device=resolved_device,
    )

    model = None
    vocoder = None
    checkpoint_path = None
    if should_generate_audio:
        checkpoint_path = resolve_checkpoint_path(config, checkpoint)
        model = SeedVCModel(config["model_params"]).to(torch_device)
        model.load_weights(checkpoint_path)
        model.eval()
        model.setup_caches(max_batch_size=1, max_seq_length=8192)
        vocoder = load_vocoder(config, torch_device)
        click.echo(f"Loaded checkpoint: {checkpoint_path}")

    items: list[dict[str, Any]] = []
    failed_count = 0

    for idx, batch in enumerate(
        tqdm(dataloader, desc="Generating results", unit="pair")
    ):
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

        status = "referenced"
        error: str | None = None

        if should_generate_audio:
            assert model is not None
            assert vocoder is not None
            ok, maybe_error = generate_audio(
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
            if ok:
                status = "generated"
            else:
                status = "generation_failed"
                error = maybe_error or "generation failed"
                failed_count += 1

        if not gen_path.exists() and status != "generation_failed":
            status = "missing_generated"
            error = "generated audio file missing"
            failed_count += 1

        item = {
            "id": idx,
            "source_path": str(src_path),
            "target_path": str(tgt_path),
            "generated_path": str(gen_path),
            "generation_status": status,
            "error": error,
            "metrics": {},
            "metric_statuses": {},
            "metric_errors": {},
        }
        items.append(item)

    generated_ok = sum(1 for item in items if item["generation_status"] == "generated")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "dataset_path": str(dataset_path),
        "config_path": str(config_path),
        "generated_base_dir": str(generated_base),
        "metric_definitions": build_metric_definitions(enable_singmos=False),
        "listening_guidance": DEFAULT_LISTENING_GUIDANCE,
        "stages": {
            "generate-results": {
                "ran_at": utc_now_iso(),
                "device": resolved_device,
                "generated_count": generated_ok,
                "failed_count": failed_count,
                "total": len(items),
                "checkpoint": checkpoint_path,
                "generated_template": generated_template,
                "generated_suffix": generated_suffix,
            }
        },
        "items": items,
    }

    save_json(output_manifest_path, manifest)
    click.echo(f"generate-results manifest written: {output_manifest_path}")
    click.echo(
        f"generate-results: total={len(items)} generated={generated_ok} failed={failed_count}"
    )
    return manifest


def run_compute_metrics_stage(
    input_manifest_path: Path,
    output_manifest_path: Path,
    device: str,
    strict: bool,
    enable_singmos: bool,
) -> dict[str, Any]:
    manifest = load_json(input_manifest_path)
    items = manifest.get("items", [])
    resolved_device = resolve_device(device)
    encoder = VoiceEncoder(device=resolved_device)
    metric_definitions = build_metric_definitions(enable_singmos=enable_singmos)
    manifest["metric_definitions"] = metric_definitions
    manifest.setdefault("listening_guidance", DEFAULT_LISTENING_GUIDANCE)

    f0_cache_root = Path(os.environ.get("CACHE_DIR", ".cache/features")) / "eval"
    f0_extractor = F0FeatureExtractor(
        features_root=f0_cache_root, device=resolved_device
    )
    embedding_cache: dict[str, np.ndarray] = {}
    f0_cache: dict[str, np.ndarray] = {}
    metric_values: dict[str, list[float]] = {
        key: []
        for key in metric_definitions
        if metric_definitions[key].get("enabled", True)
    }
    metric_status_counts: dict[str, dict[str, int]] = {
        key: {} for key in metric_definitions
    }
    singmos_predictor = None
    singmos_load_error: str | None = None
    if enable_singmos:
        try:
            singmos_predictor = load_singmos_predictor()
        except Exception as exc:  # noqa: BLE001
            singmos_load_error = str(exc)

    def _bump_status(metric_key: str, status: str) -> None:
        counts = metric_status_counts.setdefault(metric_key, {})
        counts[status] = counts.get(status, 0) + 1

    def _get_embedding(audio_path: Path) -> np.ndarray:
        cache_key = str(audio_path.resolve())
        if cache_key not in embedding_cache:
            wave = preprocess_wav(audio_path)
            embedding_cache[cache_key] = np.asarray(
                encoder.embed_utterance(wave), dtype=np.float32
            ).reshape(-1)
        return embedding_cache[cache_key]

    def _get_f0(audio_path: Path) -> np.ndarray:
        cache_key = str(audio_path.resolve())
        if cache_key not in f0_cache:
            f0_cache[cache_key] = (
                f0_extractor.extract(audio_path).detach().cpu().numpy().reshape(-1)
            )
        return f0_cache[cache_key]

    for item in tqdm(items, desc="Computing metrics", unit="pair"):
        ensure_item_metric_maps(item)
        gen_path = Path(item["generated_path"])
        src_path = Path(item["source_path"])
        tgt_path = Path(item["target_path"])

        if not gen_path.exists():
            item["error"] = f"Missing generated audio: {gen_path}"
            for metric_key, metric_meta in metric_definitions.items():
                status = (
                    "disabled"
                    if not metric_meta.get("enabled", True)
                    else "missing_generated"
                )
                set_metric_result(
                    item,
                    metric_key,
                    status=status,
                    error=item["error"] if status != "disabled" else None,
                )
                _bump_status(metric_key, status)
            if strict:
                raise FileNotFoundError(item["error"])
            continue

        try:
            tgt_emb = _get_embedding(tgt_path)
            gen_emb = _get_embedding(gen_path)
            score = cosine_similarity(tgt_emb, gen_emb)
            set_metric_result(item, DEFAULT_METRIC_KEY, status="ok", value=score)
            metric_values[DEFAULT_METRIC_KEY].append(score)
            _bump_status(DEFAULT_METRIC_KEY, "ok")
        except Exception as exc:  # noqa: BLE001
            set_metric_result(
                item,
                DEFAULT_METRIC_KEY,
                status="metric_failed",
                error=str(exc),
            )
            _bump_status(DEFAULT_METRIC_KEY, "metric_failed")
            if strict:
                raise

        try:
            source_f0 = _get_f0(src_path)
            generated_f0 = _get_f0(gen_path)
            f0_metrics = compute_f0_metrics(source_f0, generated_f0)
            for metric_key in ("f0_rmse", "f0_correlation"):
                value = f0_metrics[metric_key]
                set_metric_result(item, metric_key, status="ok", value=value)
                metric_values[metric_key].append(value)
                _bump_status(metric_key, "ok")
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            for metric_key in ("f0_rmse", "f0_correlation"):
                set_metric_result(
                    item, metric_key, status="unavailable", error=error_text
                )
                _bump_status(metric_key, "unavailable")
            if strict:
                raise

        if not metric_definitions["singmos_naturalness"].get("enabled", False):
            set_metric_result(item, "singmos_naturalness", status="disabled")
            _bump_status("singmos_naturalness", "disabled")
        elif singmos_load_error is not None:
            set_metric_result(
                item,
                "singmos_naturalness",
                status="unavailable",
                error=singmos_load_error,
            )
            _bump_status("singmos_naturalness", "unavailable")
        else:
            try:
                assert singmos_predictor is not None
                singmos_score = compute_singmos_mean(singmos_predictor, gen_path)
                set_metric_result(
                    item,
                    "singmos_naturalness",
                    status="ok",
                    value=singmos_score,
                )
                metric_values["singmos_naturalness"].append(singmos_score)
                _bump_status("singmos_naturalness", "ok")
            except Exception as exc:  # noqa: BLE001
                set_metric_result(
                    item,
                    "singmos_naturalness",
                    status="metric_failed",
                    error=str(exc),
                )
                _bump_status("singmos_naturalness", "metric_failed")
                if strict:
                    raise

        item["metric_status"] = item["metric_statuses"].get(
            DEFAULT_METRIC_KEY, "not_run"
        )
        item["error"] = item["metric_errors"].get(DEFAULT_METRIC_KEY) or item.get(
            "error"
        )

    summary_metrics = {
        key: build_metric_summary(
            key,
            metric_values.get(key, []),
            metric_status_counts.get(key, {}),
            metric_meta=metric_definitions.get(key),
        )
        for key in metric_definitions
    }
    default_summary = summary_metrics.get(DEFAULT_METRIC_KEY, {})
    summary: dict[str, Any] = {
        "default_metric": DEFAULT_METRIC_KEY,
        "metrics": summary_metrics,
        "scored": default_summary.get("scored", 0),
        "missing": default_summary.get("status_counts", {}).get("missing_generated", 0),
        "failed": default_summary.get("status_counts", {}).get("metric_failed", 0),
    }
    for key in ("mean", "median", "std", "min", "max"):
        if key in default_summary:
            summary[key] = default_summary[key]

    manifest.setdefault("stages", {})["compute-metrics"] = {
        "ran_at": utc_now_iso(),
        "device": resolved_device,
        "summary": summary,
    }

    save_json(output_manifest_path, manifest)
    click.echo(f"compute-metrics manifest written: {output_manifest_path}")
    status_parts = []
    for metric_key, metric_summary in summary_metrics.items():
        status_parts.append(f"{metric_key}=scored:{metric_summary.get('scored', 0)}")
    click.echo(f"compute-metrics: {' '.join(status_parts)}")
    return manifest


def render_report_html(
    manifest: dict[str, Any],
    report_path: Path,
    template_path: Path | None,
) -> None:
    items = manifest.get("items", [])
    metric_definitions = normalize_metric_definitions(manifest)
    summary = manifest.get("stages", {}).get("compute-metrics", {}).get("summary", {})
    summary_metrics = summary.get("metrics", {})
    rows: list[dict[str, Any]] = []
    metric_cards: list[dict[str, Any]] = []

    for metric_key, meta in metric_definitions.items():
        metric_summary = summary_metrics.get(metric_key, {})
        mean_value = metric_summary.get("mean")
        metric_cards.append(
            {
                "key": metric_key,
                "label": meta.get("label", metric_key),
                "short_label": meta.get("short_label", meta.get("label", metric_key)),
                "category": meta.get("category", "Other"),
                "description": meta.get("description"),
                "direction": meta.get("direction", "higher"),
                "enabled": meta.get("enabled", True),
                "mean": mean_value,
                "mean_display": format_metric_value(metric_key, mean_value),
                "scored": metric_summary.get("scored", 0),
                "status_counts": metric_summary.get("status_counts", {}),
            }
        )

    for item in items:
        row_metrics: dict[str, Any] = {}
        for metric_key, meta in metric_definitions.items():
            value = item.get("metrics", {}).get(metric_key)
            metric_value = float(value) if isinstance(value, (int, float)) else None
            status = item.get("metric_statuses", {}).get(metric_key)
            if status is None:
                if not meta.get("enabled", True):
                    status = "disabled"
                elif metric_value is not None:
                    status = "ok"
                else:
                    status = "not_run"
            metric_error = item.get("metric_errors", {}).get(metric_key)
            row_metrics[metric_key] = {
                "key": metric_key,
                "label": meta.get("label", metric_key),
                "short_label": meta.get("short_label", meta.get("label", metric_key)),
                "value": metric_value,
                "display": format_metric_value(metric_key, metric_value),
                "status": status,
                "error": metric_error,
                "direction": meta.get("direction", "higher"),
                "sort_value": metric_sort_value(
                    metric_value, meta.get("direction", "higher")
                ),
            }
        rows.append(
            {
                "id": item.get("id"),
                "source_path": relativize_for_report(item["source_path"], report_path),
                "target_path": relativize_for_report(item["target_path"], report_path),
                "generated_path": relativize_for_report(
                    item["generated_path"], report_path
                ),
                "generation_status": item.get("generation_status", "unknown"),
                "metrics": row_metrics,
                "error": item.get("error"),
            }
        )

    failed_rows = sum(
        1
        for row in rows
        if row["metrics"].get(DEFAULT_METRIC_KEY, {}).get("status") != "ok"
        or row["generation_status"] == "generation_failed"
    )

    default_template_path = (
        Path(__file__).resolve().parent / "templates" / "eval_report.html.j2"
    )
    chosen_template_path = template_path or default_template_path
    env = Environment(
        loader=FileSystemLoader(str(chosen_template_path.parent)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(chosen_template_path.name)

    rendered = template.render(
        generated_at=utc_now_iso(),
        default_metric_key=summary.get("default_metric", DEFAULT_METRIC_KEY),
        metrics=metric_cards,
        total_rows=len(rows),
        scored_rows=summary_metrics.get(
            summary.get("default_metric", DEFAULT_METRIC_KEY), {}
        ).get("scored", 0),
        failed_rows=failed_rows,
        listening_guidance=manifest.get(
            "listening_guidance", DEFAULT_LISTENING_GUIDANCE
        ),
        rows=rows,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered)


def run_build_report_stage(
    metrics_manifest_path: Path,
    report_path: Path,
    template_path: Path | None,
) -> dict[str, Any]:
    manifest = load_json(metrics_manifest_path)
    render_report_html(manifest, report_path, template_path)
    report_stage = {
        "ran_at": utc_now_iso(),
        "report_path": str(report_path),
        "template_path": str(template_path) if template_path else None,
    }
    manifest.setdefault("stages", {})["build-report"] = report_stage
    save_json(metrics_manifest_path, manifest)
    click.echo(f"build-report: html report written: {report_path}")
    return manifest


@click.command(context_settings={"show_default": True})
@click.option(
    "--stage",
    type=click.Choice(["all", "generate-results", "compute-metrics", "build-report"]),
    default="all",
    help="Stage to run. 'all' runs generate-results, compute-metrics, and build-report.",
)
@click.option(
    "--split",
    default="val",
    type=click.Choice(["train", "val"]),
    show_default=True,
    help="Dataset split to evaluate (reads DATA_PROCESSED/{split}.csv).",
)
@click.option(
    "--config",
    "config_path",
    default="configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Training config used for model and feature settings.",
)
@click.option(
    "--require-features/--allow-compute-missing",
    default=True,
    help="Require feature files (DATA_FEATURES), or compute missing features on demand.",
)
@click.option(
    "--generated-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory used for generated artifacts and default manifests/report.",
)
@click.option(
    "--results-manifest",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path for generate-results output manifest or compute-metrics input manifest.",
)
@click.option(
    "--metrics-manifest",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path for compute-metrics output manifest or build-report input manifest.",
)
@click.option(
    "--report-path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to output HTML evaluation report.",
)
@click.option(
    "--report-template",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional custom Jinja2 template file for the report.",
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
    help="Raise an error when a generated file is missing or sample processing fails.",
)
@click.option(
    "--enable-singmos/--disable-singmos",
    default=True,
    help="Compute SingMOS-Pro naturalness scores during compute-metrics.",
)
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Checkpoint path used by generate-results stage.",
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
    help="Generate audio during generate-results stage.",
)
def main(
    stage: str,
    split: str,
    config_path: Path,
    require_features: bool,
    generated_dir: Path | None,
    results_manifest: Path | None,
    metrics_manifest: Path | None,
    report_path: Path | None,
    report_template: Path | None,
    generated_suffix: str,
    match_field: str,
    generated_template: str | None,
    device: str,
    strict: bool,
    enable_singmos: bool,
    checkpoint: str | None,
    diffusion_steps: int,
    length_adjust: float,
    cfg_rate: float,
    pitch_shift: int,
    auto_f0_adjust: bool,
    generate: bool,
) -> None:
    import os

    dataset_path = Path(os.environ["DATA_PROCESSED"]) / f"{split}.csv"
    generated_base, resolved_results, resolved_metrics, resolved_report = (
        resolve_default_artifacts(
            dataset_path=dataset_path,
            generated_dir=generated_dir,
            results_manifest=results_manifest,
            metrics_manifest=metrics_manifest,
            report_path=report_path,
        )
    )

    if stage in {"compute-metrics", "build-report"}:
        if stage == "compute-metrics" and not resolved_results.exists():
            raise FileNotFoundError(
                f"Results manifest not found: {resolved_results}. Run --stage generate-results first."
            )
        if stage == "build-report" and not resolved_metrics.exists():
            raise FileNotFoundError(
                f"Metrics manifest not found: {resolved_metrics}. Run --stage compute-metrics first."
            )

    if stage == "generate-results":
        run_generate_results_stage(
            dataset_path=dataset_path,
            split=split,
            config_path=config_path,
            require_features=require_features,
            generated_base=generated_base,
            generated_suffix=generated_suffix,
            match_field=match_field,
            generated_template=generated_template,
            device=device,
            checkpoint=checkpoint,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            cfg_rate=cfg_rate,
            pitch_shift=pitch_shift,
            auto_f0_adjust=auto_f0_adjust,
            should_generate_audio=generate,
            output_manifest_path=resolved_results,
        )
        return

    if stage == "compute-metrics":
        run_compute_metrics_stage(
            input_manifest_path=resolved_results,
            output_manifest_path=resolved_metrics,
            device=device,
            strict=strict,
            enable_singmos=enable_singmos,
        )
        return

    if stage == "build-report":
        run_build_report_stage(
            metrics_manifest_path=resolved_metrics,
            report_path=resolved_report,
            template_path=report_template,
        )
        return

    run_generate_results_stage(
        dataset_path=dataset_path,
        split=split,
        config_path=config_path,
        require_features=require_features,
        generated_base=generated_base,
        generated_suffix=generated_suffix,
        match_field=match_field,
        generated_template=generated_template,
        device=device,
        checkpoint=checkpoint,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        cfg_rate=cfg_rate,
        pitch_shift=pitch_shift,
        auto_f0_adjust=auto_f0_adjust,
        should_generate_audio=generate,
        output_manifest_path=resolved_results,
    )
    run_compute_metrics_stage(
        input_manifest_path=resolved_results,
        output_manifest_path=resolved_metrics,
        device=device,
        strict=strict,
        enable_singmos=enable_singmos,
    )
    manifest = run_build_report_stage(
        metrics_manifest_path=resolved_metrics,
        report_path=resolved_report,
        template_path=report_template,
    )

    summary = manifest.get("stages", {}).get("compute-metrics", {}).get("summary", {})
    summary_metrics = summary.get("metrics", {})
    default_metric = summary.get("default_metric", DEFAULT_METRIC_KEY)
    default_summary = summary_metrics.get(default_metric, summary)
    if default_summary.get("scored", 0) == 0:
        click.echo("No valid samples were evaluated.")
        raise SystemExit(1)

    click.echo(
        f"{default_summary.get('label', 'Primary metric')} -> "
        f"mean: {default_summary['mean']:.4f}, "
        f"median: {default_summary['median']:.4f}, std: {default_summary['std']:.4f}, "
        f"min: {default_summary['min']:.4f}, max: {default_summary['max']:.4f}"
    )
    click.echo(f"Report: {resolved_report}")


if __name__ == "__main__":
    main()
