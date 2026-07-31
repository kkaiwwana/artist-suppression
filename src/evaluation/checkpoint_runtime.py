"""Restore an artist-control runtime from a Lightning checkpoint.

Evaluation tools must reconstruct the exact dataset vocabulary used during
training.  This module deliberately requires the Hydra config stored beside
the checkpoint instead of silently composing the repository's current config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict


def locate_checkpoint_config(
    checkpoint_path: str | Path,
    explicit_config_path: str | Path | None = None,
) -> Path:
    """Locate the resolved ``.hydra/config.yaml`` for a checkpoint run."""

    if explicit_config_path is not None:
        config_path = Path(explicit_config_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        return config_path

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for parent in checkpoint.parents:
        for relative in (
            Path("hydra_output") / ".hydra" / "config.yaml",
            Path(".hydra") / "config.yaml",
        ):
            candidate = parent / relative
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "could not locate hydra_output/.hydra/config.yaml above checkpoint "
        f"{checkpoint}; pass --config explicitly"
    )


def trusted_torch_load(path: str | Path) -> Any:
    """Load a trusted local Lightning checkpoint with a low-memory fast path."""

    checkpoint = Path(path).expanduser().resolve()
    try:
        return torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return torch.load(checkpoint, map_location="cpu", weights_only=False)


def load_checkpoint_config(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    musicgen_model_path: str | Path | None = None,
    local_files_only: bool = True,
) -> tuple[DictConfig, Path]:
    """Load the run config and apply evaluation-only runtime overrides."""

    resolved_config_path = locate_checkpoint_config(checkpoint_path, config_path)
    config = OmegaConf.load(resolved_config_path)
    dataset_cfg = config.runner.dataset
    with open_dict(dataset_cfg):
        dataset_cfg.num_workers = 0
        dataset_cfg.persistent_workers = False
        dataset_cfg.eval_num_workers = 0
        dataset_cfg.eval_persistent_workers = False
        if musicgen_model_path is not None:
            dataset_cfg.processor_name_or_path = str(
                Path(musicgen_model_path).expanduser().resolve()
            )
        dataset_cfg.local_files_only = bool(local_files_only)

    generator_cfg = config.runner.model.model_cfg.generator
    with open_dict(generator_cfg):
        if musicgen_model_path is not None:
            generator_cfg.model_name_or_path = str(
                Path(musicgen_model_path).expanduser().resolve()
            )
        # The full stage-two checkpoint already contains adapter tensors.
        # Loading the stage-one handoff again can silently overwrite them.
        generator_cfg.adapter_checkpoint = None
        if generator_cfg.get("model_kwargs") is None:
            generator_cfg.model_kwargs = {}
        if generator_cfg.get("processor_kwargs") is None:
            generator_cfg.processor_kwargs = {}
        generator_cfg.model_kwargs.local_files_only = bool(local_files_only)
        generator_cfg.processor_kwargs.local_files_only = bool(local_files_only)
    return config, resolved_config_path


def restore_control_runtime(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    musicgen_model_path: str | Path | None = None,
    local_files_only: bool = True,
    device: torch.device | str = "cpu",
):
    """Restore the exact DataModule, vocabulary, and artist-control model."""

    from src.runner import setup_dataset, setup_model

    config, resolved_config_path = load_checkpoint_config(
        checkpoint_path,
        config_path=config_path,
        musicgen_model_path=musicgen_model_path,
        local_files_only=local_files_only,
    )
    datamodule = setup_dataset(config)
    model = setup_model(config)
    payload = trusted_torch_load(checkpoint_path)
    state_dict = payload.get("state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval().to(device)
    learner = getattr(model, "concept_learner", None)
    if learner is not None and getattr(learner, "peer_mode", None) == "similarity":
        learner.refresh_peer_similarity()
    return config, resolved_config_path, datamodule, model


__all__ = [
    "load_checkpoint_config",
    "locate_checkpoint_config",
    "restore_control_runtime",
    "trusted_torch_load",
]
