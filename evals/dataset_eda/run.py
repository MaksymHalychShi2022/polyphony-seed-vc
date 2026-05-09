"""Dataset EDA report for the solo→polyphony training dataset.

Reads processed segment CSVs and optionally the raw Polyphony Project directory
and a SingMOS-Pro scores file, then renders a self-contained HTML report.

Usage:
    uv run python evals/dataset_eda/run.py
    uv run python evals/dataset_eda/run.py \\
        --raw-dir /path/to/polyphony --scores data/singmos_scores.json

Reads DATA_PROCESSED from the environment (set in .env).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import soundfile as sf
from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

TEMPLATE_PATH = Path(__file__).parent / "report.html.j2"
RESULTS_ROOT = Path(__file__).parent / "results"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
    format="%(message)s",
    level=logging.INFO,
)
log = logging.getLogger("dataset_eda")
console = Console()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_segments(processed_dir: Path) -> dict[str, list[dict]]:
    """Read train.csv and val.csv; compute durations via soundfile metadata."""
    result: dict[str, list[dict]] = {"train": [], "val": []}

    for split in ("train", "val"):
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            log.warning("Missing %s — skipping", csv_path)
            continue

        with open(csv_path, newline="") as f:
            peek = f.read(1024)
            f.seek(0)
            has_header = "source" in peek or "target" in peek
            reader = csv.DictReader(f) if has_header else csv.reader(f)
            rows = list(reader)

        pairs = []
        has_duration = has_header and "duration" in (rows[0].keys() if rows else [])
        if not has_duration:
            log.warning(
                "%s has no 'duration' column — computing durations from audio metadata (slow)",
                csv_path.name,
            )

        with Progress(
            TextColumn(f"  [cyan]{split}.csv[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("reading", total=len(rows))
            for row in rows:
                if has_header:
                    src = row.get("source", "")
                    tgt = row.get("target", "")
                else:
                    src, tgt = (row + ["", ""])[:2]

                if not src or not tgt:
                    progress.advance(task)
                    continue

                if has_duration:
                    try:
                        dur = float(row.get("duration", 0))
                    except (ValueError, TypeError):
                        dur = _audio_duration(_resolve(Path(src), processed_dir))
                else:
                    dur = _audio_duration(_resolve(Path(src), processed_dir))

                # derive song id from source path (parent directory name)
                song_id = Path(src).parent.name

                pairs.append(
                    {"source": src, "target": tgt, "duration": dur, "song_id": song_id}
                )
                progress.advance(task)

        result[split] = pairs

    return result


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def _audio_duration(path: Path) -> float:
    """Return duration in seconds using soundfile metadata (no decode)."""
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def scan_raw_dir(raw_dir: Path) -> dict[str, Any]:
    """Walk raw song subdirectories, read audio metadata per mic track."""
    songs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not songs:
        log.warning("No subdirectories found in %s", raw_dir)
        return {}

    song_data = []
    with Progress(
        TextColumn("  [cyan]raw dir scan[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("songs", total=len(songs))
        for song_dir in sorted(songs):
            tracks = [
                f for f in song_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS
            ]
            total_dur = 0.0
            for track in tracks:
                total_dur += _audio_duration(track)
            song_data.append(
                {
                    "song_id": song_dir.name,
                    "track_count": len(tracks),
                    "total_duration_s": total_dur,
                }
            )
            progress.advance(task)

    return {"songs": song_data}


def load_scores(scores_path: Path | None) -> dict[str, float] | None:
    """Load SingMOS-Pro scores; auto-detect JSON or CSV format."""
    if scores_path is None:
        return None
    if not scores_path.exists():
        log.warning("Scores file not found: %s", scores_path)
        return None

    suffix = scores_path.suffix.lower()
    try:
        if suffix == ".json":
            with open(scores_path) as f:
                data = json.load(f)
            return {str(k): float(v) for k, v in data.items()}
        else:
            result = {}
            with open(scores_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    keys = list(row.keys())
                    if len(keys) >= 2:
                        result[row[keys[0]]] = float(row[keys[1]])
            return result
    except Exception as e:
        log.warning("Failed to load scores from %s: %s", scores_path, e)
        return None


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _histogram(values: list[float], bins: int = 15) -> dict[str, Any]:
    """Return {labels, counts, rows} where rows is a list of [label, count] for templates."""
    if not values:
        return {"labels": [], "counts": [], "rows": []}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {
            "labels": [f"{lo:.1f}"],
            "counts": [len(values)],
            "rows": [[f"{lo:.1f}", len(values)]],
        }
    width = (hi - lo) / bins
    counts = [0] * bins
    labels = []
    for i in range(bins):
        labels.append(f"{lo + i * width:.1f}–{lo + (i + 1) * width:.1f}")
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    rows = [[labels[i], counts[i]] for i in range(bins)]
    return {"labels": labels, "counts": counts, "rows": rows}


def compute_segment_stats(segments: dict[str, list[dict]]) -> dict[str, Any]:
    train = segments.get("train", [])
    val = segments.get("val", [])
    all_pairs = train + val

    train_dur = [r["duration"] for r in train]
    val_dur = [r["duration"] for r in val]
    all_dur = [r["duration"] for r in all_pairs]

    # per-song pair counts
    song_counts: dict[str, int] = {}
    for r in all_pairs:
        song_counts[r["song_id"]] = song_counts.get(r["song_id"], 0) + 1

    counts = list(song_counts.values())
    median_count = statistics.median(counts) if counts else 0
    imbalanced = sorted(song_counts.items(), key=lambda x: x[1], reverse=True)
    top5 = imbalanced[:5]
    bottom5 = list(reversed(imbalanced[-5:]))
    is_imbalanced = counts and max(counts) > 3 * median_count

    dur_hist = _histogram(all_dur)
    # add per-split counts using the same bins so the chart can show train vs val
    if all_dur and len(dur_hist["labels"]) > 1:
        bins = len(dur_hist["labels"])
        lo, hi = min(all_dur), max(all_dur)
        width = (hi - lo) / bins
        train_counts = [0] * bins
        val_counts = [0] * bins
        for v in train_dur:
            train_counts[min(int((v - lo) / width), bins - 1)] += 1
        for v in val_dur:
            val_counts[min(int((v - lo) / width), bins - 1)] += 1
        dur_hist["train_counts"] = train_counts
        dur_hist["val_counts"] = val_counts

    def dur_stats(durations: list[float]) -> dict:
        if not durations:
            return {}
        return {
            "count": len(durations),
            "total_h": sum(durations) / 3600,
            "min_s": min(durations),
            "max_s": max(durations),
            "mean_s": statistics.mean(durations),
            "median_s": statistics.median(durations),
            "p10_s": _percentile(durations, 10),
            "p90_s": _percentile(durations, 90),
        }

    return {
        "total_pairs": len(all_pairs),
        "train_count": len(train),
        "val_count": len(val),
        "train_ratio": len(train) / max(len(all_pairs), 1),
        "val_ratio": len(val) / max(len(all_pairs), 1),
        "total_duration_h": sum(all_dur) / 3600,
        "train_stats": dur_stats(train_dur),
        "val_stats": dur_stats(val_dur),
        "all_stats": dur_stats(all_dur),
        "duration_histogram": dur_hist,
        "song_counts": song_counts,
        "per_song_min": min(counts) if counts else 0,
        "per_song_max": max(counts) if counts else 0,
        "per_song_mean": statistics.mean(counts) if counts else 0,
        "per_song_median": median_count,
        "top5_songs": top5,
        "bottom5_songs": bottom5,
        "is_imbalanced": is_imbalanced,
        "unique_songs": len(song_counts),
    }


def compute_raw_stats(raw_data: dict[str, Any]) -> dict[str, Any]:
    songs = raw_data.get("songs", [])
    if not songs:
        return {}

    track_counts = [s["track_count"] for s in songs]
    total_dur_s = sum(s["total_duration_s"] for s in songs)

    return {
        "total_songs": len(songs),
        "total_raw_hours": total_dur_s / 3600,
        "track_min": min(track_counts),
        "track_max": max(track_counts),
        "track_mean": statistics.mean(track_counts),
        "track_median": statistics.median(track_counts),
        "track_histogram": _histogram(
            [float(c) for c in track_counts],
            bins=max(1, max(track_counts) - min(track_counts) + 1),
        ),
        "songs": songs,
    }


def compute_filter_stats(
    raw_stats: dict[str, Any],
    segment_stats: dict[str, Any],
    scores: dict[str, float],
) -> dict[str, Any]:
    total_raw = raw_stats.get("total_songs", 0)
    songs_in_dataset = segment_stats.get("unique_songs", 0)
    kept = songs_in_dataset
    dropped = max(0, total_raw - kept)

    score_values = list(scores.values())
    kept_scores = [
        scores[sid] for sid in scores if sid in segment_stats.get("song_counts", {})
    ]
    threshold = min(kept_scores) if kept_scores else None

    return {
        "total_raw_songs": total_raw,
        "kept_songs": kept,
        "dropped_songs": dropped,
        "keep_rate": kept / max(total_raw, 1),
        "score_histogram": _histogram(score_values),
        "score_min": min(score_values) if score_values else None,
        "score_max": max(score_values) if score_values else None,
        "score_mean": statistics.mean(score_values) if score_values else None,
        "inferred_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Inline SVG charts (no CDN dependency)
# ---------------------------------------------------------------------------


def _svg_grouped_bar(
    labels: list[str],
    series: list[tuple[str, str, list[int]]],  # (name, color, counts)
    width: int = 680,
    height: int = 210,
) -> str:
    ml, mr, mt, mb = 38, 16, 16, 62
    cw = width - ml - mr
    ch = height - mt - mb

    max_val = max((c for _, _, counts in series for c in counts), default=1) or 1
    n = len(labels)
    if n == 0:
        return ""

    group_w = cw / n
    n_s = len(series)
    pad = group_w * 0.12
    bar_w = (group_w - pad * (n_s + 1)) / n_s

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
        f' style="width:100%;height:{height}px;display:block">'
    ]

    # gridlines + y-axis labels
    ticks = 4
    for i in range(ticks + 1):
        yv = max_val * i / ticks
        y = mt + ch - ch * i / ticks
        parts.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + cw}" y2="{y:.1f}"'
            f' stroke="#ddd3c6" stroke-width="{"1.5" if i == 0 else "0.8"}"/>'
        )
        parts.append(
            f'<text x="{ml - 4}" y="{y + 3.5:.1f}" text-anchor="end"'
            f' font-size="10" fill="#6b7280">{int(yv)}</text>'
        )

    # bars + x labels
    for gi, label in enumerate(labels):
        gx = ml + gi * group_w
        for si, (_, color, counts) in enumerate(series):
            val = counts[gi] if gi < len(counts) else 0
            bh = (val / max_val) * ch
            bx = gx + pad + si * (bar_w + pad)
            by = mt + ch - bh
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}"'
                f' height="{bh:.1f}" fill="{color}" rx="2"/>'
            )
        # rotated x-axis label
        lx = gx + group_w / 2
        ly = mt + ch + 6
        short = label if len(label) <= 12 else label[:11] + "…"
        parts.append(
            f'<text transform="translate({lx:.1f},{ly:.1f}) rotate(38)"'
            f' font-size="9" fill="#6b7280">{short}</text>'
        )

    # left axis line
    parts.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}"'
        f' stroke="#9ca3af" stroke-width="1.5"/>'
    )

    # legend
    legend_x = ml + cw - 90
    for si, (name, color, _) in enumerate(series):
        ly = mt + si * 18
        parts.append(
            f'<rect x="{legend_x}" y="{ly}" width="11" height="11"'
            f' fill="{color}" rx="2"/>'
        )
        parts.append(
            f'<text x="{legend_x + 15}" y="{ly + 9}" font-size="11" fill="#374151">{name}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(
    run_dir: Path,
    segment_stats: dict[str, Any],
    raw_stats: dict[str, Any],
    filter_stats: dict[str, Any] | None,
    generated_at: str,
    inputs: dict[str, str],
) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)), autoescape=True
    )
    env.filters["tojson"] = json.dumps
    template = env.get_template(TEMPLATE_PATH.name)

    dh = segment_stats.get("duration_histogram", {})
    dur_svg = (
        _svg_grouped_bar(
            dh.get("labels", []),
            [
                (
                    "Train",
                    "rgba(15,118,110,0.7)",
                    dh.get("train_counts", dh.get("counts", [])),
                ),
                ("Val", "rgba(217,119,6,0.7)", dh.get("val_counts", [])),
            ],
        )
        if dh.get("train_counts")
        else _svg_grouped_bar(
            dh.get("labels", []),
            [("All", "rgba(15,118,110,0.55)", dh.get("counts", []))],
        )
    )

    ctx = {
        "generated_at": generated_at,
        "inputs": inputs,
        "segment": segment_stats,
        "raw": raw_stats if raw_stats else None,
        "filter": filter_stats,
        "dur_svg": dur_svg,
    }

    html = template.render(**ctx)
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def print_summary(
    segment_stats: dict, raw_stats: dict, filter_stats: dict | None
) -> None:
    table = Table(title="Dataset EDA Summary", border_style="cyan", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    seg = segment_stats
    table.add_row("Total pairs", str(seg.get("total_pairs", "—")))
    table.add_row("Train pairs", str(seg.get("train_count", "—")))
    table.add_row("Val pairs", str(seg.get("val_count", "—")))
    table.add_row("Total hours", f"{seg.get('total_duration_h', 0):.2f} h")
    table.add_row("Unique songs (dataset)", str(seg.get("unique_songs", "—")))

    if raw_stats:
        table.add_row("Raw songs total", str(raw_stats.get("total_songs", "—")))
        table.add_row("Raw hours total", f"{raw_stats.get('total_raw_hours', 0):.2f} h")
        table.add_row("Tracks/song (mean)", f"{raw_stats.get('track_mean', 0):.1f}")

    if filter_stats:
        table.add_row("Songs kept", str(filter_stats.get("kept_songs", "—")))
        table.add_row("Songs dropped", str(filter_stats.get("dropped_songs", "—")))
        if filter_stats.get("inferred_threshold") is not None:
            table.add_row(
                "SingMOS threshold (min kept)",
                f"{filter_stats['inferred_threshold']:.4f}",
            )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--raw-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory of raw Polyphony Project recordings",
)
@click.option(
    "--scores",
    default=None,
    type=click.Path(path_type=Path),
    help="SingMOS-Pro scores file (JSON {song_id: score} or CSV song_id,score)",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory (default: dataset_eda/results/<timestamp>/)",
)
def main(
    raw_dir: Path | None,
    scores: Path | None,
    output_dir: Path | None,
) -> None:
    """Compute dataset statistics and render an HTML EDA report."""
    processed_dir = Path(os.environ["DATA_PROCESSED"]).expanduser().resolve()

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    if not train_csv.exists() and not val_csv.exists():
        raise click.ClickException(
            f"Neither train.csv nor val.csv found in {processed_dir}"
        )

    if raw_dir is not None and not raw_dir.exists():
        raise click.ClickException(f"Raw directory not found: {raw_dir}")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir or (RESULTS_ROOT / timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Dataset EDA[/bold cyan]")

    # load segments
    console.print("[bold]Loading segments...[/bold]")
    segments = load_segments(processed_dir)

    # scan raw dir
    raw_data: dict[str, Any] = {}
    if raw_dir is not None:
        console.print("[bold]Scanning raw directory...[/bold]")
        raw_data = scan_raw_dir(raw_dir)
    else:
        console.print("[dim]--raw-dir not provided; skipping raw statistics[/dim]")

    # load scores
    scores_data = load_scores(scores)
    if scores is not None and scores_data is None:
        console.print(
            "[dim]Scores file could not be loaded; skipping filter statistics[/dim]"
        )
    elif scores is None:
        console.print("[dim]--scores not provided; skipping filter statistics[/dim]")

    # compute stats
    console.print("[bold]Computing statistics...[/bold]")
    segment_stats = compute_segment_stats(segments)
    raw_stats = compute_raw_stats(raw_data) if raw_data else {}
    filter_stats: dict | None = None
    if scores_data is not None and raw_stats:
        filter_stats = compute_filter_stats(raw_stats, segment_stats, scores_data)

    # print terminal summary
    print_summary(segment_stats, raw_stats, filter_stats)

    # write metrics JSON
    metrics = {
        "generated_at": generated_at,
        "segment": segment_stats,
        "raw": raw_stats if raw_stats else None,
        "filter": filter_stats,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    console.print(f"[bold green]Metrics saved:[/bold green]  {metrics_path}")

    # render report
    inputs = {
        "processed_dir": str(processed_dir),
        "raw_dir": str(raw_dir) if raw_dir else None,
        "scores": str(scores) if scores else None,
    }
    report_path = render_report(
        run_dir, segment_stats, raw_stats, filter_stats, generated_at, inputs
    )

    console.print(f"[bold green]Report saved:[/bold green]   {report_path}")


if __name__ == "__main__":
    main()
