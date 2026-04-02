"""
Save training artifacts from the latest MLflow run.

Writes to the current working directory:
  - metrics.json   — final metric values (train/loss, eval/loss, ...)
  - loss_curve.png — training loss plotted over steps
"""

import json
import os
import sys


def main() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Gather the most recent run across all experiments
    experiments = client.search_experiments()
    if not experiments:
        print("No MLflow experiments found. Run training first.", file=sys.stderr)
        sys.exit(1)

    all_runs = []
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        all_runs.extend(runs)

    if not all_runs:
        print("No MLflow runs found. Run training first.", file=sys.stderr)
        sys.exit(1)

    latest_run = max(all_runs, key=lambda r: r.info.start_time)
    run_id = latest_run.info.run_id
    print(f"Using run: {run_id}")

    # ── metrics.json ─────────────────────────────────────────────────────────
    metrics = {k: float(v) for k, v in latest_run.data.metrics.items()}
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Wrote metrics.json: {metrics}")

    # ── loss_curve.png ───────────────────────────────────────────────────────
    history = client.get_metric_history(run_id, "train/loss")
    if not history:
        print("No 'train/loss' history found — loss_curve.png not written.")
        return

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    steps = [m.step for m in history]
    values = [m.value for m in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, linewidth=1.5, label="train/loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig("loss_curve.png", dpi=100)
    plt.close(fig)
    print("Wrote loss_curve.png")


if __name__ == "__main__":
    main()
