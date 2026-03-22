import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import mlflow
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TrainLogger:
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self._started = False
        self._progress_bar: tqdm | None = None

    @property
    def experiment_dir(self) -> Path:
        self._require_started()
        return self._experiment_dir

    def start(self) -> None:
        experiments_dir = os.getenv("EXPERIMENTS_DIR", "runs")
        self._experiment_dir = Path(experiments_dir) / self.experiment_name
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_path = self._experiment_dir / "train.log"

        self._logger = logging.getLogger(f"seed_vc.train.{self.experiment_name}")
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

        self.writer = SummaryWriter(str(self._experiment_dir))

        self._configure_mlflow()

        self._started = True
        self.info(f"Experiment started: {self.experiment_name}")

    def info(self, message: str) -> None:
        self._require_started()
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._require_started()
        self._logger.warning(message)

    def error(self, message: str, exc: BaseException | None = None) -> None:
        self._require_started()
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
        self._require_started()
        src = Path(path)
        dst = self._experiment_dir / src.name
        shutil.copyfile(src, dst)
        return str(dst)

    def metric(
        self,
        name: str,
        value: float,
        step: int,
        context: str | None = None,
    ) -> None:
        self._require_started()
        metric_name = f"{context}/{name}" if context else name
        self.writer.add_scalar(metric_name, value, step)
        self.writer.flush()

        mlflow.log_metric(metric_name, float(value), step=step)

    def progress(
        self,
        advance: int = 1,
        total: int | None = None,
        desc: str | None = None,
        close: bool = False,
    ) -> None:
        self._require_started()
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
        if not self._started:
            return
        self.progress(close=True)
        self.writer.close()
        mlflow.log_artifact(str(self.train_log_path))
        mlflow.end_run()
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "TrainLogger has not been started. Call logger.start() before logging."
            )

    def _configure_mlflow(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI environment variable is not set.")

        mlflow.set_tracking_uri(tracking_uri)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{self.experiment_name}_{timestamp}"
        mlflow.set_experiment(unique_name)
        mlflow.start_run(run_name=unique_name)
        self._logger.info(
            f"MLflow logging enabled for experiment '{unique_name}' at {tracking_uri}"
        )
