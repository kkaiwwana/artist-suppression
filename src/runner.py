"""Hydra-to-runtime construction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
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

    Nested generator and ConceptLearner configs are deliberately passed to the
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


__all__ = ["setup_model"]
