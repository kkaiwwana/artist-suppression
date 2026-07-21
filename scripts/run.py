"""Train explicit MusicGen artist controls with Lightning and W&B.

Run from the repository root::

    D:\\conda\\python.exe scripts/run.py exp.cmt=artist_unlearning

Set ``WANDB_API_KEY`` (preferred) or place the key in the ignored
``wandb_api_key.txt`` file.  ``RESUME_FROM`` may point to a Lightning
checkpoint or be set to ``last`` to select the newest local ``last.ckpt``.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from src.callback.audio_comparison import ValidationAudioComparisonCallback
from src.callback.git_diff import GitDiffCallback
from src.runner import setup_dataset, setup_model


log = logging.getLogger(__name__)


def _configure_windows_utf8_console() -> None:
    """Prevent Rich/W&B teardown failures on legacy GBK consoles."""

    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _read_wandb_key() -> str | None:
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key
    key_file = PROJECT_ROOT / "wandb_api_key.txt"
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
        return key or None
    return None


def _configure_wandb_paths(config: DictConfig) -> None:
    """Keep W&B staging/cache files inside the writable experiment tree."""

    runtime_dir = Path(config.exp.save_dir) / "wandb_runtime"
    defaults = {
        "WANDB_DATA_DIR": runtime_dir / "data",
        "WANDB_CACHE_DIR": runtime_dir / "cache",
        "WANDB_CONFIG_DIR": runtime_dir / "config",
    }
    for name, path in defaults.items():
        if not os.environ.get(name, "").strip():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)


def _resolve_resume_checkpoint(value: str | None, logs_dir: Path) -> str | None:
    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    if value.lower() == "last":
        candidates = sorted(
            logs_dir.glob("**/checkpoints/last.ckpt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"no last.ckpt found below {logs_dir}")
        checkpoint = candidates[0]
    else:
        checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    return str(checkpoint)


def _wandb_logger(config: DictConfig) -> WandbLogger:
    _configure_wandb_paths(config)
    wandb_cfg = config.get("wandb", {})
    mode = str(wandb_cfg.get("mode", "online"))
    os.environ["WANDB_MODE"] = mode
    key = _read_wandb_key()
    if key and mode == "online":
        wandb.login(key=key, relogin=False)
    elif mode == "online" and not key:
        log.warning(
            "WANDB_API_KEY/wandb_api_key.txt not found; relying on an existing "
            "wandb login. Set WANDB_MODE=offline for local runs."
        )

    entity = wandb_cfg.get("entity")
    run_id = wandb_cfg.get("run_id")
    logger = WandbLogger(
        project=str(config.exp.project),
        entity=None if entity in (None, "null", "") else str(entity),
        name=str(config.exp.uuid),
        id=None if run_id in (None, "null", "") else str(run_id),
        resume="allow",
        save_dir=str(config.exp.save_dir),
        tags=list(config.runner.tags),
        log_model=bool(wandb_cfg.get("log_model", False)),
        offline=mode == "offline",
    )
    logger.log_hyperparams(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )
    return logger


def _build_callbacks(config: DictConfig) -> list[pl.Callback]:
    checkpoint_cfg = config.runner.get("checkpoint", {})
    checkpoint_dir = Path(config.exp.save_dir) / "checkpoints"
    best_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:03d}-step{step}",
        monitor=str(checkpoint_cfg.get("monitor", "val/loss_total")),
        mode=str(checkpoint_cfg.get("mode", "min")),
        save_top_k=int(checkpoint_cfg.get("save_top_k", 3)),
        save_last=False,
        auto_insert_metric_name=False,
    )
    callbacks: list[pl.Callback] = [GitDiffCallback(config), best_checkpoint]
    if bool(checkpoint_cfg.get("save_last", True)):
        # Lightning 2.6 only refreshes ``save_last=True`` when a top-k model is
        # also accepted. Keep an unmonitored, fixed-name checkpoint so resume
        # never depends on a potentially noisy validation score.
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="last",
                monitor=None,
                save_top_k=1,
                save_last=False,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
                enable_version_counter=False,
                auto_insert_metric_name=False,
            )
        )
    comparison_cfg = OmegaConf.to_container(
        config.runner.get("validation_audio", {}), resolve=True
    )
    if not isinstance(comparison_cfg, dict):
        raise TypeError("runner.validation_audio must resolve to a mapping")
    callbacks.extend(
        [
            LearningRateMonitor(logging_interval="step"),
            ValidationAudioComparisonCallback(**comparison_cfg),
        ]
    )
    return callbacks


def _set_scheduler_steps(config: DictConfig, train_batches: int) -> None:
    scheduler = config.runner.get("scheduler")
    if scheduler is None or not scheduler.get("name"):
        return
    accumulation = int(config.runner.trainer.get("accumulate_grad_batches", 1))
    optimizer_steps = max(1, math.ceil(train_batches / accumulation))
    scheduler.scheduler_params.steps_per_epoch = optimizer_steps


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(config: DictConfig) -> None:
    _configure_windows_utf8_console()
    pl.seed_everything(int(config.exp.seed), workers=True)
    torch.set_float32_matmul_precision("medium")
    Path(config.exp.save_dir).mkdir(exist_ok=True, parents=True)

    datamodule = setup_dataset(config)
    _set_scheduler_steps(config, len(datamodule.train_dataloader()))
    model = setup_model(config)
    logger = _wandb_logger(config)
    callbacks = _build_callbacks(config)

    resume_value = config.exp.get("resume_from")
    checkpoint_path = _resolve_resume_checkpoint(
        resume_value,
        PROJECT_ROOT / "logs",
    )
    if checkpoint_path:
        log.info("Resuming complete Lightning state from %s", checkpoint_path)

    trainer_kwargs: dict[str, Any] = OmegaConf.to_container(
        config.runner.trainer,
        resolve=True,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(config.exp.save_dir),
        **trainer_kwargs,
    )
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
