import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TrainLogger:
    def __init__(
        self,
        experiment_dir: str,
        tensorboard_writer: SummaryWriter | None = None,
        mlflow_client: Any | None = None,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_path = self.experiment_dir / "train.log"

        self._logger = logging.getLogger(f"seed_vc.train.{self.experiment_dir.name}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(self.train_log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(stream_handler)

        self.writer = tensorboard_writer or SummaryWriter(str(self.experiment_dir))
        self._mlflow = mlflow_client
        self._mlflow_run_started = False
        self._configure_mlflow()

        self._progress_bar: tqdm | None = None

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str, exc: BaseException | None = None) -> None:
        if exc is not None:
            self._logger.error(
                "%s: %s",
                message,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return

        has_exc = sys.exc_info()[0] is not None
        self._logger.error(message, exc_info=has_exc)

    def save_artifact(self, path: str) -> str:
        src = Path(path)
        dst = self.experiment_dir / src.name
        shutil.copyfile(src, dst)
        return str(dst)

    def metric(
        self,
        name: str,
        value: float,
        step: int,
        context: str | None = None,
    ) -> None:
        metric_name = f"{context}/{name}" if context else name
        self.writer.add_scalar(metric_name, value, step)
        self.writer.flush()

        if self._mlflow is None:
            return

        try:
            self._mlflow.log_metric(metric_name, float(value), step=step)
        except Exception as exc:
            self.warning(f"Failed to log metric '{metric_name}' to MLflow: {exc}")

    def progress(
        self,
        advance: int = 1,
        total: int | None = None,
        desc: str | None = None,
        close: bool = False,
    ) -> None:
        if close:
            if self._progress_bar is not None:
                self._progress_bar.close()
                self._progress_bar = None
            return

        if self._progress_bar is None:
            if total is None:
                raise ValueError("total must be provided when initializing progress")
            self._progress_bar = tqdm(total=total, desc=desc)
            if advance == 0:
                return

        if desc and self._progress_bar is not None:
            self._progress_bar.set_description(desc)
        if self._progress_bar is not None and advance:
            self._progress_bar.update(advance)

    def close(self) -> None:
        self.progress(close=True)
        self.writer.close()
        if self._mlflow is not None and self._mlflow_run_started:
            try:
                self._mlflow.end_run()
            except Exception as exc:
                self.warning(f"Failed to close MLflow run: {exc}")
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

    def _configure_mlflow(self) -> None:
        if self._mlflow is not None:
            return

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            return

        try:
            import mlflow  # type: ignore

            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = self.experiment_dir.name
            mlflow.set_experiment(experiment_name)
            if mlflow.active_run() is None:
                mlflow.start_run(run_name=experiment_name)
                self._mlflow_run_started = True
            self._mlflow = mlflow
            self.info(
                f"MLflow logging enabled for experiment '{experiment_name}' at {tracking_uri}"
            )
        except Exception as exc:
            self._mlflow = None
            self.warning(
                f"MLflow disabled: failed to initialize remote tracking: {exc}"
            )
