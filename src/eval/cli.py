import json
import os
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

from seed_vc.train.features_dataset import build_features_dataloader
from seed_vc.train.seed_vc_model import SeedVCModel
from seed_vc.utils.hf_utils import load_custom_model_from_hf

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

MANIFEST_SCHEMA_VERSION = 1
METRIC_KEY = "resemblyzer_similarity"
DEFAULT_RESULTS_MANIFEST_NAME = "results_manifest.json"
DEFAULT_METRICS_MANIFEST_NAME = "metrics_manifest.json"
DEFAULT_REPORT_NAME = "evaluation_report.html"


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
    config_path: Path,
    cache_root: Path,
    require_cache: bool,
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
        }
        items.append(item)

    generated_ok = sum(1 for item in items if item["generation_status"] == "generated")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "dataset_path": str(dataset_path),
        "config_path": str(config_path),
        "generated_base_dir": str(generated_base),
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
) -> dict[str, Any]:
    manifest = load_json(input_manifest_path)
    items = manifest.get("items", [])
    resolved_device = resolve_device(device)
    encoder = VoiceEncoder(device=resolved_device)

    scores: list[float] = []
    failures = 0
    missing = 0

    for item in tqdm(items, desc="Computing metrics", unit="pair"):
        gen_path = Path(item["generated_path"])
        tgt_path = Path(item["target_path"])

        if not gen_path.exists():
            item["metric_status"] = "missing_generated"
            item["error"] = f"Missing generated audio: {gen_path}"
            missing += 1
            if strict:
                raise FileNotFoundError(item["error"])
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
            score = cosine_similarity(tgt_emb, gen_emb)
            item.setdefault("metrics", {})[METRIC_KEY] = score
            item["metric_status"] = "ok"
            if item.get("error") and item["generation_status"] != "generation_failed":
                item["error"] = None
            scores.append(score)
        except Exception as exc:  # noqa: BLE001
            item["metric_status"] = "metric_failed"
            item["error"] = str(exc)
            failures += 1
            if strict:
                raise

    scored = len(scores)
    summary: dict[str, Any] = {
        "metric": METRIC_KEY,
        "scored": scored,
        "missing": missing,
        "failed": failures,
    }
    if scored > 0:
        arr = np.asarray(scores)
        summary.update(
            {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )

    manifest.setdefault("stages", {})["compute-metrics"] = {
        "ran_at": utc_now_iso(),
        "device": resolved_device,
        "summary": summary,
    }

    save_json(output_manifest_path, manifest)
    click.echo(f"compute-metrics manifest written: {output_manifest_path}")
    click.echo(f"compute-metrics: scored={scored} missing={missing} failed={failures}")
    return manifest


def render_report_html(
    manifest: dict[str, Any],
    report_path: Path,
    template_path: Path | None,
) -> None:
    items = manifest.get("items", [])
    rows: list[dict[str, Any]] = []
    numeric_scores: list[float] = []

    for item in items:
        value = item.get("metrics", {}).get(METRIC_KEY)
        metric_value = float(value) if isinstance(value, (int, float)) else None
        if metric_value is not None:
            numeric_scores.append(metric_value)
        rows.append(
            {
                "id": item.get("id"),
                "source_path": relativize_for_report(item["source_path"], report_path),
                "target_path": relativize_for_report(item["target_path"], report_path),
                "generated_path": relativize_for_report(
                    item["generated_path"], report_path
                ),
                "metric_value": metric_value,
                "metric_display": f"{metric_value:.4f}"
                if metric_value is not None
                else "N/A",
                "generation_status": item.get("generation_status", "unknown"),
                "metric_status": item.get("metric_status", "not_run"),
                "error": item.get("error"),
            }
        )

    mean_metric = float(np.mean(np.asarray(numeric_scores))) if numeric_scores else None
    failed_rows = sum(
        1
        for row in rows
        if row["metric_status"] != "ok"
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
        metric_name=METRIC_KEY,
        mean_metric=mean_metric,
        total_rows=len(rows),
        scored_rows=len(numeric_scores),
        failed_rows=failed_rows,
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
    dataset_path: Path,
    config_path: Path,
    cache_root: Path,
    require_cache: bool,
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
    checkpoint: str | None,
    diffusion_steps: int,
    length_adjust: float,
    cfg_rate: float,
    pitch_shift: int,
    auto_f0_adjust: bool,
    generate: bool,
) -> None:
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
            config_path=config_path,
            cache_root=cache_root,
            require_cache=require_cache,
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
        config_path=config_path,
        cache_root=cache_root,
        require_cache=require_cache,
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
    )
    manifest = run_build_report_stage(
        metrics_manifest_path=resolved_metrics,
        report_path=resolved_report,
        template_path=report_template,
    )

    summary = manifest.get("stages", {}).get("compute-metrics", {}).get("summary", {})
    if summary.get("scored", 0) == 0:
        click.echo("No valid samples were evaluated.")
        raise SystemExit(1)

    click.echo(
        f"Resemblyzer cosine similarity -> mean: {summary['mean']:.4f}, "
        f"median: {summary['median']:.4f}, std: {summary['std']:.4f}, "
        f"min: {summary['min']:.4f}, max: {summary['max']:.4f}"
    )
    click.echo(f"Report: {resolved_report}")


if __name__ == "__main__":
    main()
