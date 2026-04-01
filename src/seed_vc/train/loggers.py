import abc
import os


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def log_metric(self, name: str, value: float, step: int) -> None: ...

    def finalize(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    def __init__(self, save_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._tb = SummaryWriter(save_dir)

    def log_metric(self, name: str, value: float, step: int) -> None:
        self._tb.add_scalar(name, value, step)
        self._tb.flush()

    def finalize(self) -> None:
        self._tb.close()


class MLflowLogger(BaseLogger):
    def __init__(self, experiment_name: str, tracking_uri: str | None = None) -> None:
        import mlflow

        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if not uri:
            raise ValueError(
                "MLflowLogger requires a tracking_uri or MLFLOW_TRACKING_URI env var."
            )
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=experiment_name)

    def log_metric(self, name: str, value: float, step: int) -> None:
        import mlflow

        mlflow.log_metric(name, float(value), step=step)

    def finalize(self) -> None:
        import mlflow

        mlflow.end_run()


class MultiLogger:
    def __init__(self, loggers: list[BaseLogger]) -> None:
        self._loggers = loggers

    def log_metric(self, name: str, value: float, step: int) -> None:
        for lg in self._loggers:
            lg.log_metric(name, value, step)

    def finalize(self) -> None:
        for lg in self._loggers:
            lg.finalize()
