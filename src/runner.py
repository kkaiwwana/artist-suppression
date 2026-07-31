"""Hydra-to-runtime construction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from queue import Empty, Queue
from threading import Thread
from typing import Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, open_dict
from torch import nn


def _instantiate_metrics(metric_config: Any):
    """Instantiate metric config while preserving caller-provided modules."""

    if metric_config is None:
        return None
    if isinstance(metric_config, nn.Module):
        return metric_config
    if isinstance(metric_config, (list, tuple, ListConfig)):
        return [
            metric if isinstance(metric, nn.Module) else instantiate(metric)
            for metric in metric_config
        ]
    if isinstance(metric_config, (dict, DictConfig, Mapping)):
        if "_target_" in metric_config:
            return instantiate(metric_config)
        return {
            str(name): metric if isinstance(metric, nn.Module) else instantiate(metric)
            for name, metric in metric_config.items()
        }
    raise TypeError(
        "runner.metrics must be an instantiated metric, list, mapping, or Hydra config"
    )


def setup_model(config: DictConfig) -> pl.LightningModule:
    """Build the configured Lightning model from the composed root config.

    Nested generator and explicit ConceptLearner configs are passed to the
    LightningModule without recursive instantiation; each LightningModule owns
    construction of its PyTorch components as required by the project design.
    """

    if config is None or "runner" not in config:
        raise ValueError("setup_model expects the composed config with runner section")
    runner_cfg = config.runner
    if "model" not in runner_cfg:
        raise ValueError("config.runner.model is required")
    metric_cfg = runner_cfg.get("metrics", runner_cfg.get("metric"))
    metrics = _instantiate_metrics(metric_cfg)
    model = instantiate(
        runner_cfg.model,
        metrics=metrics,
        optimizer_cfg=runner_cfg.optimizer,
        scheduler_cfg=runner_cfg.get("scheduler"),
        # Passing the complete config as an instantiate override makes Hydra
        # eagerly resolve runtime-only values such as ${hydra:runtime.cwd}.
        # A config produced by the Compose API does not necessarily have a
        # global HydraConfig, so attach it after construction instead.
        cfg=None,
        _recursive_=False,
    )
    if not isinstance(model, pl.LightningModule):
        raise TypeError(
            "config.runner.model must instantiate pytorch_lightning.LightningModule, "
            f"got {type(model).__name__}"
        )
    model.cfg = config
    return model


def setup_dataset(config: DictConfig) -> pl.LightningDataModule:
    """Instantiate/setup the DataModule and resolve explicit concept metadata."""

    if config is None or "runner" not in config:
        raise ValueError("setup_dataset expects the composed config with runner section")
    dataset_cfg = config.runner.get("dataset")
    if dataset_cfg is None:
        raise ValueError("config.runner.dataset is required")
    datamodule = instantiate(dataset_cfg)
    if not isinstance(datamodule, pl.LightningDataModule):
        raise TypeError(
            "config.runner.dataset must instantiate a LightningDataModule, "
            f"got {type(datamodule).__name__}"
        )
    datamodule.setup("fit")

    model_cfg = config.runner.get("model")
    learner_cfg = None
    if model_cfg is not None:
        nested_model_cfg = model_cfg.get("model_cfg")
        if nested_model_cfg is not None:
            learner_cfg = nested_model_cfg.get("concept_learner")
    if learner_cfg is not None and hasattr(datamodule, "num_concepts"):
        expected = int(datamodule.num_concepts)
        configured = learner_cfg.get("num_concepts")
        with open_dict(learner_cfg):
            if configured in (None, "auto"):
                learner_cfg["num_concepts"] = expected
            elif int(configured) != expected:
                raise ValueError(
                    f"ConceptLearner num_concepts={configured} but the DataModule "
                    f"built {expected}; set num_concepts: auto"
                )
            granularity = str(getattr(datamodule, "concept_granularity", "artist"))
            if learner_cfg.get("concept_name") in (None, "auto"):
                learner_cfg["concept_name"] = granularity
            if granularity != "artist":
                raise ValueError(
                    "explicit copyright control currently requires "
                    "concept_granularity=artist"
                )
            # Genre metadata is only injected for an explicit legacy ablation.
            # Similarity-centred control derives peers from decoded residuals.
            if str(learner_cfg.get("peer_mode", "similarity")) == "genre":
                group_matrix = getattr(datamodule, "artist_genre_matrix", None)
                if group_matrix is not None and learner_cfg.get(
                    "group_matrix"
                ) in (None, "auto"):
                    learner_cfg["group_matrix"] = group_matrix.tolist()

    # Resolve classifier output size and validation-loader names only after
    # the raw-audio DataModule has reconstructed the checkpoint vocabulary.
    if model_cfg is not None:
        nested_model_cfg = model_cfg.get("model_cfg")
        classifier_cfg = (
            nested_model_cfg.get("classifier")
            if nested_model_cfg is not None
            else None
        )
        if classifier_cfg is not None and hasattr(datamodule, "num_classes"):
            expected_classes = int(datamodule.num_classes)
            configured_classes = classifier_cfg.get("num_classes")
            with open_dict(classifier_cfg):
                if configured_classes in (None, "auto"):
                    classifier_cfg["num_classes"] = expected_classes
                elif int(configured_classes) != expected_classes:
                    raise ValueError(
                        f"classifier num_classes={configured_classes} but the "
                        f"DataModule built {expected_classes} classes"
                    )
            if nested_model_cfg.get("validation_names") in (None, "auto"):
                with open_dict(nested_model_cfg):
                    nested_model_cfg["validation_names"] = list(
                        getattr(datamodule, "validation_names", ("gt",))
                    )
    return datamodule


def timeout_input(timeout: float = 10.0) -> int | None:
    """Read one console line without blocking checkpoint resume forever."""

    queue: Queue[str] = Queue(maxsize=1)

    def _read() -> None:
        try:
            queue.put(input())
        except (EOFError, KeyboardInterrupt):
            pass

    Thread(target=_read, daemon=True).start()
    try:
        value = queue.get(timeout=timeout).strip()
        return int(value) if value else None
    except (Empty, ValueError):
        return None


__all__ = ["setup_dataset", "setup_model", "timeout_input"]
