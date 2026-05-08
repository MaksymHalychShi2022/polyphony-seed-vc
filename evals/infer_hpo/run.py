"""Inference hyperparameter grid search for Seed-VC.

Evaluates a cartesian product of diffusion_steps × cfg_rate combinations,
generates audio, computes eval metrics, and produces a ranked HTML report.

Usage:
    uv run python evals/infer_hpo/run.py --config <cfg> --checkpoint <ckpt>
"""

from __future__ import annotations

import itertools
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import torch
import yaml
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

os_environ_patch = __import__("os").environ
os_environ_patch.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")

from eval.cli import (  # noqa: E402
    DEFAULT_METRIC_KEY,
    MANIFEST_SCHEMA_VERSION,
    build_metric_definitions,
    generate_audio,
    load_json,
    load_vocoder,
    resolve_device,
    run_compute_metrics_stage,
    save_json,
    utc_now_iso,
)
from seed_vc.modules.vc_wrapper import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CHECKPOINT_REPO_ID,
)
from seed_vc.train.features_dataset import build_features_dataloader  # noqa: E402
from seed_vc.train.seed_vc_model import SeedVCModel  # noqa: E402
from seed_vc.utils.hf_utils import load_custom_model_from_hf  # noqa: E402


def resolve_checkpoint_path(checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    log.info(
        f"No checkpoint given — using default: {DEFAULT_CHECKPOINT_REPO_ID}/{DEFAULT_CHECKPOINT}"
    )
    ckpt = load_custom_model_from_hf(
        DEFAULT_CHECKPOINT_REPO_ID, DEFAULT_CHECKPOINT, None
    )
    return ckpt[0] if isinstance(ckpt, tuple) else ckpt


TEMPLATE_PATH = Path(__file__).parent / "report.html.j2"
RESULTS_ROOT = Path(__file__).parent / "results"
MAX_CACHED_BATCHES_WARN = 200

log = logging.getLogger("infer_hpo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def combo_slug(steps: int, cfg_rate: float) -> str:
    rate_str = f"{cfg_rate:.2f}".replace(".", "p")
    return f"steps{steps}_cfg{rate_str}"


def build_combo_grid(
    diffusion_steps: list[int],
    cfg_rates: list[float],
    length_adjust: float,
    auto_f0_adjust: bool,
    pitch_shift: int,
) -> list[dict[str, Any]]:
    combos = []
    for steps, cfg in itertools.product(diffusion_steps, cfg_rates):
        combos.append(
            {
                "slug": combo_slug(steps, cfg),
                "diffusion_steps": steps,
                "cfg_rate": cfg,
                "length_adjust": length_adjust,
                "auto_f0_adjust": auto_f0_adjust,
                "pitch_shift": pitch_shift,
            }
        )
    return combos


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)

    rich_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    rich_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )

    log.addHandler(rich_handler)
    log.addHandler(file_handler)


def print_dry_run_table(combos: list[dict[str, Any]]) -> None:
    console = Console()
    table = Table(title="Grid search combos (dry run)", show_lines=True)
    table.add_column("#", style="dim")
    table.add_column("slug")
    table.add_column("diffusion_steps", justify="right")
    table.add_column("cfg_rate", justify="right")
    table.add_column("length_adjust", justify="right")
    table.add_column("auto_f0_adjust")
    table.add_column("pitch_shift", justify="right")
    for i, c in enumerate(combos, 1):
        table.add_row(
            str(i),
            c["slug"],
            str(c["diffusion_steps"]),
            f"{c['cfg_rate']:.2f}",
            f"{c['length_adjust']:.2f}",
            str(c["auto_f0_adjust"]),
            str(c["pitch_shift"]),
        )
    console.print(table)
    console.print(f"\n[bold]{len(combos)} combos total.[/bold]")


def extract_metric_means(metrics_manifest: dict[str, Any]) -> dict[str, float | None]:
    summary = (
        metrics_manifest.get("stages", {}).get("compute-metrics", {}).get("summary", {})
    )
    summary_metrics = summary.get("metrics", {})
    return {
        key: summary_metrics.get(key, {}).get("mean")
        for key in (
            "resemblyzer_similarity",
            "f0_rmse",
            "f0_correlation",
            "singmos_naturalness",
        )
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def run_grid_search(
    run_dir: Path,
    combos: list[dict[str, Any]],
    batches: list[tuple],
    model: SeedVCModel,
    vocoder,
    device: torch.device,
    sample_rate: int,
    dataset_path: Path,
    config_path: Path,
    checkpoint_path: str,
    resume: bool,
    progress: Progress,
) -> list[dict[str, Any]]:
    combo_task = progress.add_task("[cyan]Combos", total=len(combos))
    results: list[dict[str, Any]] = []

    for combo in combos:
        slug = combo["slug"]
        combo_dir = run_dir / "combos" / slug

        if resume and (combo_dir / "metrics_manifest.json").exists():
            log.info(f"[resume] skipping {slug} — metrics_manifest.json already exists")
            metrics_manifest = load_json(combo_dir / "metrics_manifest.json")
            means = extract_metric_means(metrics_manifest)
            lat = metrics_manifest.get("grid_search_latency", {})
            results.append(
                {
                    "slug": slug,
                    "params": {k: v for k, v in combo.items() if k != "slug"},
                    "metrics": means,
                    "generation_seconds": lat.get("generation_seconds", 0.0),
                    "metrics_seconds": lat.get("metrics_seconds", 0.0),
                    "total_seconds": lat.get("total_seconds", 0.0),
                    "generation_seconds_per_sample": lat.get(
                        "generation_seconds_per_sample", 0.0
                    ),
                    "generated_count": lat.get("generated_count", 0),
                    "total_samples": lat.get("total_samples", 0),
                }
            )
            progress.advance(combo_task)
            continue

        combo_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"Starting combo {slug} "
            f"(steps={combo['diffusion_steps']}, cfg={combo['cfg_rate']:.2f})"
        )

        # --- generation ---
        sample_task = progress.add_task(f"[yellow]  {slug}", total=len(batches))
        items: list[dict[str, Any]] = []
        failed = 0
        gen_start = time.perf_counter()

        for idx, batch in enumerate(batches):
            (
                src_mels,
                src_mel_lengths,
                tgt_mels,
                tgt_mel_lengths,
                src_semantics,
                _,
                tgt_semantics,
                _,
                src_f0s,
                _,
                tgt_f0s,
                _,
                tgt_embeddings,
                src_paths,
                tgt_paths,
            ) = batch

            src_path = Path(src_paths[0])
            tgt_path = Path(tgt_paths[0])

            pair_dir = run_dir / "tracks" / f"{src_path.stem}__{tgt_path.stem}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            # copy source/target once per pair
            for orig, dest_name in ((src_path, "source.wav"), (tgt_path, "target.wav")):
                dest = pair_dir / dest_name
                if not dest.exists() and orig.exists():
                    shutil.copy2(orig, dest)

            gen_path = pair_dir / f"{slug}.wav"
            ok, err = generate_audio(
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
                device=device,
                diffusion_steps=combo["diffusion_steps"],
                length_adjust=combo["length_adjust"],
                cfg_rate=combo["cfg_rate"],
                auto_f0_adjust=combo["auto_f0_adjust"],
                pitch_shift=combo["pitch_shift"],
                sample_rate=sample_rate,
            )

            if not ok:
                failed += 1

            items.append(
                {
                    "id": idx,
                    "source_path": str(src_path),
                    "target_path": str(tgt_path),
                    "generated_path": str(gen_path),
                    "generation_status": "generated" if ok else "generation_failed",
                    "error": err,
                    "metrics": {},
                    "metric_statuses": {},
                    "metric_errors": {},
                }
            )
            progress.advance(sample_task)

        gen_seconds = time.perf_counter() - gen_start
        generated_ok = len(items) - failed

        progress.remove_task(sample_task)
        log.info(
            f"{slug} generation done: {generated_ok}/{len(items)} ok "
            f"in {gen_seconds:.1f}s ({gen_seconds / max(len(items), 1):.2f}s/sample)"
        )

        # --- write results manifest ---
        results_manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "dataset_path": str(dataset_path),
            "config_path": str(config_path),
            "generated_base_dir": str(combo_dir),
            "metric_definitions": build_metric_definitions(enable_singmos=False),
            "stages": {
                "generate-results": {
                    "ran_at": utc_now_iso(),
                    "device": str(device),
                    "generated_count": generated_ok,
                    "failed_count": failed,
                    "total": len(items),
                    "checkpoint": checkpoint_path,
                }
            },
            "items": items,
        }
        results_path = combo_dir / "results_manifest.json"
        save_json(results_path, results_manifest)

        # --- metrics ---
        metrics_path = combo_dir / "metrics_manifest.json"
        met_start = time.perf_counter()
        metrics_manifest = run_compute_metrics_stage(
            input_manifest_path=results_path,
            output_manifest_path=metrics_path,
            device=str(device),
            strict=False,
            enable_singmos=True,
        )
        met_seconds = time.perf_counter() - met_start

        means = extract_metric_means(metrics_manifest)
        total_seconds = gen_seconds + met_seconds

        # store latency in the metrics manifest for resume
        metrics_manifest["grid_search_latency"] = {
            "generation_seconds": gen_seconds,
            "metrics_seconds": met_seconds,
            "total_seconds": total_seconds,
            "generation_seconds_per_sample": gen_seconds / max(len(items), 1),
            "generated_count": generated_ok,
            "total_samples": len(items),
        }
        save_json(metrics_path, metrics_manifest)

        log.info(
            f"{slug} metrics done in {met_seconds:.1f}s | "
            f"resemblyzer={means.get('resemblyzer_similarity')!r} "
            f"f0_rmse={means.get('f0_rmse')!r} "
            f"f0_corr={means.get('f0_correlation')!r}"
        )

        results.append(
            {
                "slug": slug,
                "params": {k: v for k, v in combo.items() if k != "slug"},
                "metrics": means,
                "generation_seconds": gen_seconds,
                "metrics_seconds": met_seconds,
                "total_seconds": total_seconds,
                "generation_seconds_per_sample": gen_seconds / max(len(items), 1),
                "generated_count": generated_ok,
                "total_samples": len(items),
            }
        )
        progress.advance(combo_task)

    return results


def aggregate_and_report(
    run_dir: Path,
    results: list[dict[str, Any]],
    grid_params: dict[str, Any],
    checkpoint_path: str,
    config_path: Path,
) -> None:
    # sort by resemblyzer_similarity descending, None last
    def sort_key(r: dict[str, Any]) -> float:
        v = r["metrics"].get(DEFAULT_METRIC_KEY)
        return v if v is not None else -1.0

    ranked = sorted(results, key=sort_key, reverse=True)
    for rank, r in enumerate(ranked, 1):
        r["rank"] = rank

    grid_manifest = {
        "created_at": utc_now_iso(),
        "checkpoint": checkpoint_path,
        "config_path": str(config_path),
        "grid_params": grid_params,
        "total_combos": len(ranked),
        "combos": ranked,
    }
    save_json(run_dir / "grid_manifest.json", grid_manifest)

    # render HTML report
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(TEMPLATE_PATH.name)
    report_html = template.render(
        generated_at=utc_now_iso(),
        combos=ranked,
        grid_params=grid_params,
        checkpoint=checkpoint_path,
    )
    (run_dir / "report.html").write_text(report_html)

    # print best combo
    if ranked:
        best = ranked[0]
        m = best["metrics"]
        log.info(
            f"Best: steps={best['params']['diffusion_steps']} "
            f"cfg={best['params']['cfg_rate']:.2f} → "
            f"resemblyzer={m.get('resemblyzer_similarity')!r}, "
            f"f0_rmse={m.get('f0_rmse')!r}, "
            f"f0_corr={m.get('f0_correlation')!r}"
        )
        log.info(f"Report: {run_dir / 'report.html'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",")]


def parse_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",")]


@click.command(context_settings={"show_default": True})
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model config YAML.",
)
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Checkpoint path. Defaults to pretrained_model from config.",
)
@click.option(
    "--split",
    default="val",
    type=click.Choice(["train", "val"]),
    help="Dataset split to evaluate.",
)
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Inference device.",
)
@click.option(
    "--diffusion-steps",
    "diffusion_steps_str",
    default="4,8,10,16,25",
    help="Comma-separated diffusion step counts to search.",
)
@click.option(
    "--cfg-rates",
    "cfg_rates_str",
    default="0.5,0.7,0.9",
    help="Comma-separated CFG rates to search.",
)
@click.option(
    "--length-adjust",
    default=1.0,
    type=float,
    help="Fixed length adjustment factor.",
)
@click.option(
    "--auto-f0-adjust/--no-auto-f0-adjust",
    default=True,
    help="Enable automatic F0 adjustment.",
)
@click.option(
    "--pitch-shift",
    default=0,
    type=int,
    help="Fixed pitch shift in semitones.",
)
@click.option(
    "--require-features/--allow-compute-missing",
    default=True,
    help="Require precomputed feature cache or compute on demand.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip combos whose metrics_manifest.json already exists.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print combo grid and exit without loading model or generating audio.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Override the run output directory (default: evals/infer_hpo/results/<timestamp>).",
)
def main(
    config_path: Path,
    checkpoint: str | None,
    split: str,
    device: str,
    diffusion_steps_str: str,
    cfg_rates_str: str,
    length_adjust: float,
    auto_f0_adjust: bool,
    pitch_shift: int,
    require_features: bool,
    resume: bool,
    dry_run: bool,
    output_dir: Path | None,
) -> None:
    import os

    diffusion_steps = parse_ints(diffusion_steps_str)
    cfg_rates = parse_floats(cfg_rates_str)

    combos = build_combo_grid(
        diffusion_steps, cfg_rates, length_adjust, auto_f0_adjust, pitch_shift
    )

    if dry_run:
        print_dry_run_table(combos)
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir or (RESULTS_ROOT / timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir / "run.log")
    log.info(f"Run directory: {run_dir}")
    log.info(
        f"Grid: {len(combos)} combos ({len(diffusion_steps)} steps × {len(cfg_rates)} cfg_rates)"
    )

    config = yaml.safe_load(config_path.read_text())
    preprocess_params = config["preprocess_params"]
    sr = int(preprocess_params.get("sr", 22050))
    spect_params = preprocess_params["spect_params"]
    whisper_model_name = config["model_params"]["speech_tokenizer"]["name"]

    resolved_device = resolve_device(device)
    torch_device = torch.device(resolved_device)

    dataset_path = Path(os.environ["DATA_PROCESSED"]) / f"{split}.csv"

    log.info("Loading dataloader and caching batches...")
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
    batches = list(dataloader)
    if len(batches) > MAX_CACHED_BATCHES_WARN:
        log.warning(
            f"Val set has {len(batches)} batches (>{MAX_CACHED_BATCHES_WARN}). "
            "Caching all in memory may be large."
        )
    log.info(f"Cached {len(batches)} val batches.")

    log.info("Resolving checkpoint (will download from HuggingFace if not cached)...")
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    log.info(f"Checkpoint resolved: {checkpoint_path}")
    log.info("Building model structure and moving to device...")
    model = SeedVCModel(config["model_params"]).to(torch_device)
    log.info("Loading weights from checkpoint (may take 30-60s for large files)...")
    model.load_weights(checkpoint_path)
    log.info("Setting up KV caches...")
    model.eval()
    model.setup_caches(max_batch_size=1, max_seq_length=8192)
    log.info("Loading vocoder...")
    vocoder = load_vocoder(config, torch_device)
    log.info("Model and vocoder ready.")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        results = run_grid_search(
            run_dir=run_dir,
            combos=combos,
            batches=batches,
            model=model,
            vocoder=vocoder,
            device=torch_device,
            sample_rate=sr,
            dataset_path=dataset_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            resume=resume,
            progress=progress,
        )

    grid_params = {
        "diffusion_steps": diffusion_steps,
        "cfg_rates": cfg_rates,
        "length_adjust": length_adjust,
        "auto_f0_adjust": auto_f0_adjust,
        "pitch_shift": pitch_shift,
    }
    aggregate_and_report(
        run_dir=run_dir,
        results=results,
        grid_params=grid_params,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
