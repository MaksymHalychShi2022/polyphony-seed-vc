"""Build a standalone comparison bundle for two cached evaluation runs."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
RUN_A_DIR = ROOT / "data/processed/data/processed/.eval_cache/20260329-174351"
RUN_B_DIR = ROOT / "data/processed/data/processed/.eval_cache/20260329-172119"
OUTPUT_DIR = ROOT / "combined-eval-report"
DEFAULT_METRIC_KEY = "resemblyzer_similarity"
TEMPLATE_PATH = ROOT / "src/eval/templates/eval_comparison_report.html.j2"
METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "resemblyzer_similarity": {
        "label": "Resemblyzer similarity",
        "short_label": "Timbre",
        "category": "Timbre similarity",
        "description": "Cosine similarity between target and generated Resemblyzer embeddings.",
        "direction": "higher",
        "decimals": 4,
    },
    "f0_rmse": {
        "label": "F0 RMSE",
        "short_label": "F0 RMSE",
        "category": "Melody preservation",
        "description": "Root-mean-square error between aligned source and generated F0 contours.",
        "direction": "lower",
        "decimals": 3,
    },
    "f0_correlation": {
        "label": "F0 correlation",
        "short_label": "F0 corr",
        "category": "Melody preservation",
        "description": "Pearson correlation between aligned source and generated F0 contours.",
        "direction": "higher",
        "decimals": 4,
    },
    "singmos_naturalness": {
        "label": "SingMOS naturalness",
        "short_label": "SingMOS",
        "category": "Naturalness",
        "description": "Mean SingMOS-Pro score across generated-audio chunks.",
        "direction": "higher",
        "decimals": 4,
    },
}


@dataclass(frozen=True)
class RunConfig:
    slug: str
    label: str
    cache_dir: Path

    @property
    def metrics_manifest_path(self) -> Path:
        return self.cache_dir / "metrics_manifest.json"

    @property
    def results_manifest_path(self) -> Path:
        return self.cache_dir / "results_manifest.json"


RUNS = (
    RunConfig(slug="run_a", label="Run A", cache_dir=RUN_A_DIR),
    RunConfig(slug="run_b", label="Run B", cache_dir=RUN_B_DIR),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def sanitize_part(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def rel_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path_str
    return path.resolve()


def canonical_input_key(path_str: str) -> str:
    return resolve_input_path(path_str).as_posix()


def build_item_key(item: dict[str, Any]) -> str:
    source = resolve_input_path(item["source_path"])
    target = resolve_input_path(item["target_path"])
    return f"{rel_to_root(source)}::{rel_to_root(target)}"


def normalize_metric_definitions(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("metric_definitions") or {}
    if not raw:
        return {
            DEFAULT_METRIC_KEY: {
                "key": DEFAULT_METRIC_KEY,
                **METRIC_DEFINITIONS[DEFAULT_METRIC_KEY],
            }
        }

    normalized: dict[str, dict[str, Any]] = {}
    for key, meta in raw.items():
        payload = dict(METRIC_DEFINITIONS.get(key, {}))
        payload.update(meta)
        payload["key"] = key
        payload.setdefault("label", key)
        payload.setdefault("short_label", payload["label"])
        payload.setdefault("direction", "higher")
        payload.setdefault("decimals", 4)
        normalized[key] = payload
    return normalized


def get_metric_summary_map(manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("stages", {}).get("compute-metrics", {}).get("summary", {})
    if "metrics" in summary:
        return summary["metrics"]
    if summary:
        return {DEFAULT_METRIC_KEY: summary}
    return {}


def format_metric_value(metric_key: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    decimals = int(METRIC_DEFINITIONS.get(metric_key, {}).get("decimals", 4))
    return f"{value:.{decimals}f}"


def metric_sort_value(metric_key: str, value: float | None) -> float | None:
    if value is None:
        return None
    direction = METRIC_DEFINITIONS.get(metric_key, {}).get("direction", "higher")
    return value if direction == "higher" else -value


def validate_run_inputs(run: RunConfig) -> None:
    if not run.cache_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run.cache_dir}")

    for manifest_path in (run.metrics_manifest_path, run.results_manifest_path):
        if not manifest_path.is_file():
            raise SystemExit(f"Required manifest not found: {manifest_path}")

    manifest = load_json(run.metrics_manifest_path)
    items = manifest.get("items", [])
    if not items:
        raise SystemExit(f"No evaluation items found in: {run.metrics_manifest_path}")

    for item in items:
        for field in ("source_path", "target_path"):
            candidate = resolve_input_path(item[field])
            if not candidate.is_file():
                raise SystemExit(
                    f"Missing referenced audio for {run.slug}: {field} -> {candidate}"
                )


def convert_audio_to_mp3(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-b:a",
            "192k",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dest.relative_to(OUTPUT_DIR).as_posix()


def copy_audio_asset(
    path_str: str,
    category: str,
    filename_hint: str,
    asset_map: dict[str, str],
) -> str | None:
    input_key = canonical_input_key(path_str)
    src = Path(input_key)
    if not src.is_file():
        return None

    existing = asset_map.get(input_key)
    if existing is not None:
        return existing

    source_parent = sanitize_part(src.parent.name)
    filename_stem = Path(filename_hint).stem
    filename = f"{source_parent}__{sanitize_part(filename_stem)}.mp3"
    dest = OUTPUT_DIR / "audio" / category / filename
    copied = convert_audio_to_mp3(src, dest)
    asset_map[input_key] = copied
    return copied


def summarize_run_metrics(
    manifest: dict[str, Any], metric_keys: list[str]
) -> dict[str, Any]:
    generate_stage = manifest.get("stages", {}).get("generate-results", {})
    summary_metrics = get_metric_summary_map(manifest)
    metrics: dict[str, Any] = {}
    for metric_key in metric_keys:
        metric_summary = summary_metrics.get(metric_key, {})
        metrics[metric_key] = {
            "mean": metric_summary.get("mean"),
            "mean_display": format_metric_value(metric_key, metric_summary.get("mean")),
            "scored": metric_summary.get("scored", 0),
            "status_counts": metric_summary.get("status_counts", {}),
        }
    return {
        "checkpoint": generate_stage.get("checkpoint"),
        "generated_count": generate_stage.get("generated_count"),
        "total": generate_stage.get("total"),
        "metrics": metrics,
    }


def build_run_entry(
    run: RunConfig,
    item: dict[str, Any],
    asset_map: dict[str, str],
    metric_keys: list[str],
) -> dict[str, Any]:
    generated_src = Path(item["generated_path"])
    generated_path = copy_audio_asset(
        item["generated_path"],
        f"{run.slug}/generated",
        generated_src.name,
        asset_map,
    )
    metrics: dict[str, Any] = {}
    metric_statuses = item.get("metric_statuses", {})
    metric_errors = item.get("metric_errors", {})
    raw_metrics = item.get("metrics", {})
    for metric_key in metric_keys:
        value = raw_metrics.get(metric_key)
        metric_value = float(value) if isinstance(value, (int, float)) else None
        status = metric_statuses.get(metric_key)
        if status is None:
            status = "ok" if metric_value is not None else "not_run"
        metrics[metric_key] = {
            "value": metric_value,
            "display": format_metric_value(metric_key, metric_value),
            "status": status,
            "error": metric_errors.get(metric_key),
            "sort_value": metric_sort_value(metric_key, metric_value),
        }
    return {
        "generation_status": item.get("generation_status", "unknown"),
        "generated_path": generated_path,
        "metrics": metrics,
    }


def rewrite_manifest_for_bundle(
    manifest: dict[str, Any],
    run: RunConfig,
    asset_map: dict[str, str],
) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(manifest))
    rewritten["generated_base_dir"] = f"audio/{run.slug}/generated"

    build_report = rewritten.get("stages", {}).get("build-report")
    if build_report is not None:
        build_report["report_path"] = "evaluation_comparison.html"

    for item in rewritten.get("items", []):
        item["source_path"] = asset_map.get(
            canonical_input_key(item["source_path"]), item["source_path"]
        )
        item["target_path"] = asset_map.get(
            canonical_input_key(item["target_path"]), item["target_path"]
        )
        item["generated_path"] = asset_map.get(
            canonical_input_key(item["generated_path"]), item["generated_path"]
        )

    return rewritten


def build_bundle() -> dict[str, Any]:
    for run in RUNS:
        validate_run_inputs(run)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict[str, Any]] = {}
    result_manifests: dict[str, dict[str, Any]] = {}
    comparison_rows: dict[str, dict[str, Any]] = {}
    asset_map: dict[str, str] = {}
    metric_definitions: dict[str, dict[str, Any]] = {}

    for run in RUNS:
        metrics_manifest = load_json(run.metrics_manifest_path)
        results_manifest = load_json(run.results_manifest_path)
        manifests[run.slug] = metrics_manifest
        result_manifests[run.slug] = results_manifest
        for key, meta in normalize_metric_definitions(metrics_manifest).items():
            metric_definitions.setdefault(key, meta)

    metric_keys = [key for key in METRIC_DEFINITIONS if key in metric_definitions] + [
        key for key in metric_definitions if key not in METRIC_DEFINITIONS
    ]

    run_summaries: list[dict[str, Any]] = []
    for run in RUNS:
        run_summaries.append(
            {
                "slug": run.slug,
                "label": run.label,
                "cache_name": run.cache_dir.name,
                "metrics_manifest": f"manifests/{run.slug}/metrics_manifest.json",
                "results_manifest": f"manifests/{run.slug}/results_manifest.json",
                **summarize_run_metrics(manifests[run.slug], metric_keys),
            }
        )

        for item in manifests[run.slug].get("items", []):
            item_key = build_item_key(item)
            source_path = Path(item["source_path"])
            target_path = Path(item["target_path"])
            row = comparison_rows.setdefault(
                item_key,
                {
                    "item_key": item_key,
                    "source_name": source_path.name,
                    "target_name": target_path.name,
                    "source_label": source_path.stem,
                    "target_label": target_path.stem,
                    "source_path": None,
                    "target_path": None,
                    "runs": {},
                },
            )

            if row["source_path"] is None:
                row["source_path"] = copy_audio_asset(
                    item["source_path"],
                    "shared/source",
                    f"{source_path.parent.name}_{source_path.name}",
                    asset_map,
                )
            if row["target_path"] is None:
                row["target_path"] = copy_audio_asset(
                    item["target_path"],
                    "shared/target",
                    f"{target_path.parent.name}_{target_path.name}",
                    asset_map,
                )

            row["runs"][run.slug] = build_run_entry(run, item, asset_map, metric_keys)

    rows: list[dict[str, Any]] = []
    for index, item_key in enumerate(sorted(comparison_rows), start=1):
        row = comparison_rows[item_key]
        run_a = row["runs"].get("run_a")
        run_b = row["runs"].get("run_b")
        deltas: dict[str, Any] = {}
        for metric_key in metric_keys:
            a_value = run_a["metrics"][metric_key]["value"] if run_a else None
            b_value = run_b["metrics"][metric_key]["value"] if run_b else None
            delta_value = None
            if a_value is not None and b_value is not None:
                delta_value = float(b_value - a_value)
            deltas[metric_key] = {
                "value": delta_value,
                "display": format_metric_value(metric_key, delta_value),
            }
        rows.append(
            {
                "index": index,
                "item_key": item_key,
                "source_name": row["source_name"],
                "target_name": row["target_name"],
                "source_label": row["source_label"],
                "target_label": row["target_label"],
                "source_path": row["source_path"],
                "target_path": row["target_path"],
                "run_a": run_a,
                "run_b": run_b,
                "deltas": deltas,
                "has_both_runs": run_a is not None and run_b is not None,
            }
        )

    for run in RUNS:
        save_json(
            OUTPUT_DIR / "manifests" / run.slug / "metrics_manifest.json",
            rewrite_manifest_for_bundle(manifests[run.slug], run, asset_map),
        )
        save_json(
            OUTPUT_DIR / "manifests" / run.slug / "results_manifest.json",
            rewrite_manifest_for_bundle(result_manifests[run.slug], run, asset_map),
        )

    comparison_manifest = {
        "generated_at": utc_now_iso(),
        "default_metric_key": DEFAULT_METRIC_KEY,
        "metrics": [
            {
                "key": key,
                "label": metric_definitions[key].get("label", key),
                "short_label": metric_definitions[key].get(
                    "short_label", metric_definitions[key].get("label", key)
                ),
                "category": metric_definitions[key].get("category", "Other"),
                "description": metric_definitions[key].get("description"),
                "direction": metric_definitions[key].get("direction", "higher"),
            }
            for key in metric_keys
        ],
        "output_dir": OUTPUT_DIR.name,
        "runs": run_summaries,
        "rows": rows,
    }
    save_json(OUTPUT_DIR / "comparison_manifest.json", comparison_manifest)
    return comparison_manifest


def render_report(comparison_manifest: dict[str, Any]) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(TEMPLATE_PATH.name)
    report_path = OUTPUT_DIR / "evaluation_comparison.html"
    report_path.write_text(template.render(**comparison_manifest))
    return report_path


def main() -> None:
    comparison_manifest = build_bundle()
    report_path = render_report(comparison_manifest)
    print(f"Standalone comparison bundle created at: {OUTPUT_DIR}")
    print(f"Comparison manifest: {OUTPUT_DIR / 'comparison_manifest.json'}")
    print(f"HTML report: {report_path}")


if __name__ == "__main__":
    main()
