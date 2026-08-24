"""Backend-neutral experiment logging for PyTorch Lightning callbacks.

The project uses :mod:`pytorch_lightning`, while recent SwanLab releases build
their bundled logger against the separate :mod:`lightning.pytorch` package.
``SwanLabLightningLogger`` mirrors SwanLab's official integration on top of the
logger base class already used by this project, avoiding two incompatible
Lightning class hierarchies in one process.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
import numbers
import re
from typing import Any, Optional
import warnings

from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn


class SwanLabLightningLogger(Logger):
    """Minimal SwanLab logger compatible with ``pytorch_lightning``.

    Initialization and metric/media calls intentionally follow SwanLab's
    official ``SwanLabLogger`` integration. SwanLab remains an optional
    dependency and is imported only when this backend is selected.
    """

    tracking_backend = "swanlab"

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        workspace: Optional[str] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        log_dir: Optional[str] = None,
        logdir: Optional[str] = None,
        mode: Optional[str] = None,
        save_dir: Optional[str] = ".",
        tags: Optional[list[str]] = None,
        id: Optional[str] = None,
        resume: Optional[str | bool] = None,
        public: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if logdir is not None:
            warnings.warn(
                "The `logdir` parameter is deprecated; use `log_dir` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            log_dir = logdir
        tags = list(tags or [])
        if "⚡️pytorch lightning" not in tags:
            tags.append("⚡️pytorch lightning")
        self._experiment: Any | None = None
        self._save_dir = None if save_dir is None else str(save_dir)
        self._init_kwargs: dict[str, Any] = {}
        values = {
            "project": project,
            "workspace": workspace,
            "experiment_name": experiment_name,
            "description": description,
            "log_dir": log_dir,
            "mode": mode,
            "tags": tags,
            "id": id,
            "resume": resume,
            "public": public,
        }
        self._init_kwargs.update(
            {name: value for name, value in values.items() if value is not None}
        )
        self._init_kwargs.update(kwargs)

    @staticmethod
    def _swanlab() -> Any:
        try:
            import swanlab
        except ImportError as error:  # pragma: no cover - depends on deployment
            raise ModuleNotFoundError(
                "logging.backend=swanlab requires the 'swanlab' package; "
                "install it with `pip install swanlab`"
            ) from error
        return swanlab

    @classmethod
    def _active_run(cls) -> Any | None:
        swanlab = cls._swanlab()
        try:
            return swanlab.get_run()
        except RuntimeError:
            return None

    @property
    def name(self) -> str:
        return "swanlab"

    @property
    def version(self) -> Optional[str]:
        run = self._active_run()
        if run is None:
            run = self._experiment
        return None if run is None else getattr(run, "id", None)

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def log_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> Any:
        if self._experiment is not None:
            return self._experiment
        swanlab = self._swanlab()
        active = self._active_run()
        if active is not None:
            rank_zero_warn(
                "An active SwanLab run already exists; reusing it for the "
                "PyTorch Lightning logger. Call swanlab.finish() first to "
                "start a separate run."
            )
            self._experiment = active
            return active
        try:
            swanlab.config["FRAMEWORK"] = "pytorch_lightning"
        except Exception:  # pragma: no cover - optional metadata only
            pass
        self._experiment = swanlab.init(**self._init_kwargs)
        return self._experiment

    @staticmethod
    def _params_to_dict(params: Any) -> dict[str, Any]:
        if isinstance(params, Namespace):
            params = vars(params)
        elif hasattr(params, "__dict__") and not isinstance(params, Mapping):
            params = vars(params)
        elif not isinstance(params, Mapping):
            return {}
        result: dict[str, Any] = {}
        for key, value in dict(params).items():
            if callable(value):
                value = getattr(value, "__name__", str(value))
            result[str(key)] = value
        return result

    @rank_zero_only
    def update_config(self, config: Mapping[str, Any]) -> None:
        self.experiment.config.update(dict(config))

    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.update_config(self._params_to_dict(params))

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: Optional[int] = None,
    ) -> None:
        self.experiment.log(dict(metrics), step=step)

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._log_media("Image", key, images, step, **kwargs)

    @rank_zero_only
    def log_audio(
        self,
        key: str,
        audios: list[Any],
        step: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._log_media("Audio", key, audios, step, **kwargs)

    @rank_zero_only
    def log_text(
        self,
        key: str,
        texts: list[Any],
        step: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._log_media("Text", key, texts, step, **kwargs)

    def _log_media(
        self,
        factory_name: str,
        key: str,
        values: list[Any],
        step: Optional[int],
        **kwargs: Any,
    ) -> None:
        if not isinstance(values, list):
            raise TypeError(f"{key!r} media values must be a list")
        factory = getattr(self._swanlab(), factory_name)
        count = len(values)
        for name, option in kwargs.items():
            if isinstance(option, (list, tuple)) and len(option) != count:
                raise ValueError(
                    f"expected {count} values for {name!r}, got {len(option)}"
                )
        media = []
        for index, value in enumerate(values):
            options = {
                name: option[index]
                if isinstance(option, (list, tuple))
                else option
                for name, option in kwargs.items()
            }
            media.append(factory(value, **options))
        self.log_metrics({key: media}, step=step)

    @rank_zero_only
    def save(self) -> None:
        return None

    @rank_zero_only
    def finalize(self, status: Optional[str] = None) -> None:
        if status is None or status == "success":
            return
        if self._active_run() is not None:
            self._swanlab().finish(
                state="crashed",
                error=f"Closed by PyTorch Lightning with status {status!r}",
            )


class _SwanLabTable:
    """W&B-like table builder converted to ``swanlab.echarts.Table`` on log."""

    def __init__(self, *, columns: list[Any]) -> None:
        self.columns = list(columns)
        self.data: list[list[Any]] = []

    def add_data(self, *values: Any) -> None:
        if len(values) != len(self.columns):
            raise ValueError(
                f"table row has {len(values)} values for {len(self.columns)} columns"
            )
        self.data.append(list(values))


class _SwanLabMediaFacade:
    """Expose SwanLab media with the small W&B-style surface callbacks use."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def Table(self, *, columns: list[Any]) -> _SwanLabTable:  # noqa: N802
        return _SwanLabTable(columns=columns)

    def Audio(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
        return self._module.Audio(*args, **kwargs)

    def Image(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
        return self._module.Image(*args, **kwargs)


def _is_table_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, numbers.Number)):
        return True
    ndim = getattr(value, "ndim", None)
    numel = getattr(value, "numel", None)
    return ndim == 0 and callable(numel) and int(numel()) == 1


def _table_scalar(value: Any) -> Any:
    if _is_table_scalar(value) and hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _metric_slug(value: Any) -> str:
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")
    return slug or "media"


@dataclass(frozen=True)
class ExperimentTracker:
    """A run plus backend media constructors used by project callbacks."""

    backend: str
    experiment: Any
    media: Any

    @classmethod
    def wandb(cls, experiment: Any, module: Any | None = None) -> "ExperimentTracker":
        if module is None:
            import wandb as module
        return cls("wandb", experiment, module)

    @classmethod
    def swanlab(
        cls,
        experiment: Any,
        module: Any | None = None,
    ) -> "ExperimentTracker":
        if module is None:
            import swanlab as module
        return cls("swanlab", experiment, _SwanLabMediaFacade(module))

    def define_metric(self, name: str, **kwargs: Any) -> None:
        if self.backend != "wandb":
            return
        define = getattr(self.experiment, "define_metric", None)
        if callable(define):
            define(name, **kwargs)

    def _swanlab_table_payload(
        self,
        key: str,
        table: _SwanLabTable,
    ) -> dict[str, Any]:
        swanlab = self.media._module
        media_columns: list[int] = []
        for column in range(len(table.columns)):
            if any(
                not _is_table_scalar(row[column])
                for row in table.data
            ):
                media_columns.append(column)
        scalar_columns = [
            column
            for column in range(len(table.columns))
            if column not in media_columns
        ]
        native = swanlab.echarts.Table()
        native.add(
            [table.columns[column] for column in scalar_columns],
            [
                [_table_scalar(row[column]) for column in scalar_columns]
                for row in table.data
            ],
        )
        payload: dict[str, Any] = {key: native}
        for column in media_columns:
            media_values = [
                row[column]
                for row in table.data
                if row[column] is not None
            ]
            if media_values:
                payload[f"{key}/{_metric_slug(table.columns[column])}"] = media_values
        return payload

    def log(self, payload: Mapping[str, Any], step: Optional[int] = None) -> None:
        if self.backend == "wandb":
            kwargs = {} if step is None else {"step": step}
            self.experiment.log(dict(payload), **kwargs)
            return
        converted: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, _SwanLabTable):
                converted.update(self._swanlab_table_payload(str(key), value))
            else:
                converted[str(key)] = value
        self.experiment.log(converted, step=step)


def experiment_tracker(trainer: Any) -> ExperimentTracker | None:
    """Return the supported experiment tracker owned by a Lightning trainer."""

    loggers = getattr(trainer, "loggers", None)
    if not loggers:
        logger = getattr(trainer, "logger", None)
        loggers = [] if logger is None else [logger]
    for logger in loggers:
        if isinstance(logger, SwanLabLightningLogger):
            return ExperimentTracker.swanlab(logger.experiment)
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:  # pragma: no cover - optional dependency
            WandbLogger = ()  # type: ignore[assignment]
        if isinstance(logger, WandbLogger):
            return ExperimentTracker.wandb(logger.experiment)
    return None


__all__ = [
    "ExperimentTracker",
    "SwanLabLightningLogger",
    "experiment_tracker",
]
