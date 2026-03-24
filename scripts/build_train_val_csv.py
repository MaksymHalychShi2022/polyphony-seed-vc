"""Build train/val CSVs from processed chunked dataset.

Expected input layout (produced by scripts/process_raw_dataset.py):
  PROCESSED_DIR/<song_id>/
    <chunk_id>_<stem_name>.flac
    <chunk_id>_mixture.flac

Output:
  TRAIN_CSV
  VAL_CSV

Each CSV has exactly two columns:
  source,target
"""

import csv
import os
import random
from pathlib import Path


import click


def list_song_dirs(processed_dir: Path) -> list[Path]:
    if not processed_dir.exists() or not processed_dir.is_dir():
        raise SystemExit(
            f"PROCESSED_DIR does not exist or is not a directory: {processed_dir}"
        )
    song_dirs = [p for p in processed_dir.iterdir() if p.is_dir()]
    song_dirs.sort(key=lambda p: p.name)
    return song_dirs


def collect_pairs(processed_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for song_dir in list_song_dirs(processed_dir):
        mixtures = sorted(song_dir.glob("*_mixture.flac"))
        for mixture_path in mixtures:
            name = mixture_path.name
            suffix = "_mixture.flac"
            if not name.endswith(suffix):
                continue
            chunk_id = name[: -len(suffix)]
            for stem_path in sorted(song_dir.glob(f"{chunk_id}_*.flac")):
                if stem_path.name == mixture_path.name:
                    continue
                pairs.append((stem_path, mixture_path))

    pairs.sort(key=lambda x: (x[1].as_posix(), x[0].as_posix()))
    return pairs


def make_relative(path: Path, csv_parent: Path) -> str:
    rel = os.path.relpath(path.resolve(), start=csv_parent.resolve())
    return Path(rel).as_posix()


def write_csv(csv_path: Path, rows: list[tuple[Path, Path]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base = csv_path.parent
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for source_path, target_path in rows:
            writer.writerow(
                [
                    make_relative(source_path, base),
                    make_relative(target_path, base),
                ]
            )


@click.command(context_settings={"show_default": True})
@click.option(
    "--val-ratio",
    type=float,
    default=0.1,
    help="Fraction of pairs to put in val.csv",
)
@click.option("--seed", type=int, default=42, help="Shuffle seed")
@click.option(
    "--max-pairs",
    type=int,
    default=-1,
    help="Limit total number of pairs before splitting (-1 = all)",
)
def main(
    val_ratio: float,
    seed: int,
    max_pairs: int,
) -> None:
    processed_dir = Path(os.environ["DATA_PROCESSED"]).expanduser().resolve()
    train_csv = (processed_dir / "train.csv").resolve()
    val_csv = (processed_dir / "val.csv").resolve()

    if not (0.0 <= float(val_ratio) <= 1.0):
        raise click.ClickException("--val-ratio must be between 0.0 and 1.0")

    pairs = collect_pairs(processed_dir)
    if not pairs:
        raise click.ClickException(
            f"No source/target pairs found in processed dataset: {processed_dir}"
        )

    rng = random.Random(int(seed))
    rng.shuffle(pairs)

    if int(max_pairs) > 0:
        pairs = pairs[: int(max_pairs)]

    total = len(pairs)
    n_val = int(total * float(val_ratio))
    val_rows = pairs[:n_val]
    train_rows = pairs[n_val:]

    write_csv(train_csv, train_rows)
    write_csv(val_csv, val_rows)

    click.echo(
        f"Done. Total pairs: {total}. Train: {len(train_rows)}. Val: {len(val_rows)}."
    )
    click.echo(f"Wrote: {train_csv}")
    click.echo(f"Wrote: {val_csv}")


if __name__ == "__main__":
    main()
