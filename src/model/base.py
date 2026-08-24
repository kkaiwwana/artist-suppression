"""Lightning wrapper for adapter fine-tuning of the pure PyTorch MusicGen.

Loss computation remains in :class:`src.model.gen.MusciGen`.  This module only
coordinates Lightning lifecycle, synchronized logging and optimizer/scheduler
construction from Hydra configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from src.model.utils import ExperimentSyncLogger


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_kwargs(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        result = OmegaConf.to_container(config, resolve=True)
    else:
        result = dict(config)
    if not isinstance(result, dict):
        raise TypeError("configuration parameters must resolve to a mapping")
    return result


def _instantiate_component(config_or_module: Any) -> nn.Module:
    if isinstance(config_or_module, nn.Module):
        return config_or_module
    if config_or_module is None:
        raise ValueError("a module instance or Hydra _target_ configuration is required")
    component = instantiate(config_or_module)
    if not isinstance(component, nn.Module):
        raise TypeError(
            "Hydra component configuration must instantiate torch.nn.Module, "
            f"got {type(component).__name__}"
        )
    return component


class BaseGenerationModel(pl.LightningModule):
    """LightningModule for ordinary MusicGen adapter fine-tuning.

    Args:
        metrics: Already-instantiated torchmetrics module, mapping, or sequence.
            Metrics are registered for checkpoint/device movement but are not
            logged during training yet.
        model_cfg: Configuration containing ``generator``, or an already
            instantiated generator module.
        optimizer_cfg: Hydra optimizer configuration with ``name`` and
            ``optimizer_params``.
        scheduler_cfg: Optional scheduler configuration with ``name``,
            ``scheduler_params`` and Lightning ``interval``/``frequency``.
        cfg: The full composed Hydra config, retained for downstream use.
        sync_logger: Optional pre-created :class:`ExperimentSyncLogger`, useful in
            tests or custom launchers.
        log_backend: Legacy fallback backend used only with an explicitly
            supplied ``sync_logger``. Normal training uses Lightning's logger.
    """

    TEST_TEXT_KEYS = (
        "text",
        "texts",
        "prompt",
        "prompts",
        "description",
        "descriptions",
    )

    def __init__(
        self,
        metrics: Optional[Any],
        model_cfg: Any,
        optimizer_cfg: Any,
        scheduler_cfg: Optional[Any] = None,
        cfg: Optional[Any] = None,
        sync_logger: Optional[ExperimentSyncLogger] = None,
        log_backend: str = "wandb",
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.cfg = cfg
        self.log_backend = log_backend
        self._sync_logger = sync_logger

        self.metrics = self._register_metrics(metrics)
        self.model = self.setup_model()
        self.test_generation_cfg = _cfg_get(
            self.model_cfg,
            "test_generation",
            {},
        )

    @staticmethod
    def _register_metrics(metrics: Optional[Any]) -> nn.Module:
        if metrics is None:
            return nn.ModuleDict()
        if isinstance(metrics, nn.Module):
            return metrics
        if isinstance(metrics, Mapping):
            return nn.ModuleDict({str(name): metric for name, metric in metrics.items()})
        if isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
            return nn.ModuleList(list(metrics))
        raise TypeError(
            "metrics must be an instantiated torch.nn.Module, mapping, sequence, or None"
        )

    def setup_model(self) -> nn.Module:
        generator_cfg = _cfg_get(self.model_cfg, "generator", self.model_cfg)
        return _instantiate_component(generator_cfg)

    def forward(self, batch: Optional[Mapping[str, Any]] = None, **kwargs):
        return self.model(batch, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def call_model_train(self, batch: Mapping[str, Any], batch_idx: int = 0):
        return self.model.training_step(batch, batch_idx)

    def call_model_eval(self, batch: Mapping[str, Any], batch_idx: int = 0):
        return self.model.validation_step(batch, batch_idx)

    def _log_losses(
        self,
        total_loss: Tensor,
        loss_dict: Mapping[str, Tensor],
        *,
        stage: str,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive when logging losses")
        on_step = stage == "train"
        payload = {f"{stage}/loss_total": total_loss.detach().float().clone()}
        for name, value in loss_dict.items():
            if name.endswith("_monitor"):
                # Put diagnostics in their own top-level W&B section. Using
                # ``train/monitor_*`` made W&B place dozens of per-block
                # diagnostics beside the actual training-loss panels.
                metric_name = (
                    f"monitor/{stage}/{name.removesuffix('_monitor')}"
                )
            elif name.startswith("branch_"):
                metric_name = f"{stage}/{name}"
            else:
                metric_name = f"{stage}/loss_{name}"
            payload[metric_name] = value.detach().float().clone()
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            self.log_dict(
                payload,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            # Stable checkpoint/progress aliases; detailed curves remain under
            # the grouped W&B names above.
            if stage == "val":
                self.log(
                    "val_loss",
                    total_loss.detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            return

        # Legacy/test fallback when no Lightning Trainer owns logging.
        if self._sync_logger is not None:
            self._sync_logger.log(
                payload,
                on_step=on_step,
                on_epoch=True,
                step=None,
            )
            if stage == "train":
                self._sync_logger.add_step()

    def _shared_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        *,
        stage: str,
    ):
        result = (
            self.call_model_train(batch, batch_idx)
            if stage == "train"
            else self.call_model_eval(batch, batch_idx)
        )
        total_loss = result["loss"]
        self._log_losses(
            total_loss,
            {"musicgen_lm": result["loss"]},
            stage=stage,
            batch_size=int(result["sample_losses"].shape[0])
            if isinstance(result.get("sample_losses"), Tensor)
            else int(len(batch.get("text", [])) or 1),
        )
        return total_loss, result

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        total_loss, _ = self._shared_step(batch, batch_idx, stage="train")
        return total_loss

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Tensor]:
        del dataloader_idx
        total_loss, _ = self._shared_step(batch, batch_idx, stage="val")
        return {"loss": total_loss.detach()}

    @classmethod
    def _test_prompts(cls, batch: Mapping[str, Any]) -> Optional[Any]:
        for key in cls.TEST_TEXT_KEYS:
            if key in batch and batch[key] is not None:
                return batch[key]
        return None

    def _generate_test_batch(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Optional[Any]]:
        """Generate audio from the common Lightning test-batch contract."""

        prompts = self._test_prompts(batch)
        generation_inputs = batch.get("generation_inputs")
        if generation_inputs is not None and not isinstance(
            generation_inputs,
            Mapping,
        ):
            raise TypeError("generation_inputs must be a mapping when supplied")
        if prompts is None and generation_inputs is None:
            raise ValueError(
                "test_step requires text/prompts (including their aliases) or "
                "preprocessed generation_inputs"
            )

        generation_kwargs = _config_kwargs(self.test_generation_cfg)
        batch_generation_kwargs = batch.get("generation_kwargs")
        if batch_generation_kwargs is not None:
            generation_kwargs.update(_config_kwargs(batch_generation_kwargs))
        audio_values = self.model.generate(
            text=prompts,
            inputs=generation_inputs,
            **generation_kwargs,
        )
        if not isinstance(audio_values, Tensor):
            raise TypeError("the generator test path must return an audio Tensor")
        return audio_values, prompts

    def _test_output(
        self,
        audio_values: Tensor,
        prompts: Optional[Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "audio_values": audio_values.detach(),
            "sample_rate": int(getattr(self.model, "audio_sample_rate", 32000)),
            "batch_idx": int(batch_idx),
        }
        if prompts is not None:
            output["prompts"] = prompts
        return output

    @torch.no_grad()
    def generate_validation_samples(
        self,
        batch: Mapping[str, Any],
        *,
        generation_kwargs: Optional[Mapping[str, Any]] = None,
        generation_seed: int = 42,
    ) -> Dict[str, Any]:
        """Decode fixed references and generate default samples for monitoring.

        This is the adapter-only counterpart to the explicit controller's
        multi-branch comparison. It deliberately evaluates only the ordinary
        no-control path, so W&B can expose teacher-forcing/free-generation
        divergence during dataset adaptation.
        """

        audio_tokens = batch.get("audio_tokens")
        if not isinstance(audio_tokens, Tensor):
            raise ValueError("validation samples require audio_tokens")
        decode = getattr(self.model, "decode_audio_tokens", None)
        if not callable(decode):
            raise RuntimeError("the generator cannot decode reference audio tokens")
        ground_truth, ground_truth_lengths = decode(
            audio_tokens,
            batch.get("decoder_attention_mask"),
        )

        generation_batch = dict(batch)
        generation_batch["generation_kwargs"] = dict(generation_kwargs or {})
        device = audio_tokens.device
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index if device.index is not None else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(generation_seed))
            generated, prompts = self._generate_test_batch(generation_batch)
        return {
            "ground_truth": ground_truth,
            "ground_truth_lengths": ground_truth_lengths,
            "generated": generated,
            "prompts": prompts,
            "metadata": batch.get("metadata"),
            "sample_rate": int(getattr(self.model, "audio_sample_rate", 32000)),
        }

    @torch.no_grad()
    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        """Run text-conditioned generation without logging or file I/O.

        Keeping audio in the returned mapping lets a Lightning callback decide
        whether to save WAV files, compute metrics, or publish media to a
        logger, while this module remains independent of those policies.
        """

        del dataloader_idx
        audio_values, prompts = self._generate_test_batch(batch)
        return self._test_output(audio_values, prompts, batch_idx)

    def set_sync_logger(self, logger: Optional[ExperimentSyncLogger]) -> None:
        self._sync_logger = logger

    def on_train_start(self) -> None:
        # The backend Lightning logger is owned by Trainer. The optional legacy
        # sync logger is
        # only used when a caller explicitly injects one (mainly old tests).
        return

    def on_validation_epoch_end(self) -> None:
        if self._sync_logger is None or getattr(self, "_trainer", None) is not None:
            return
        self._sync_logger.add_epoch()
        self._sync_logger.log({}, on_epoch=True)
        self._sync_logger.sync_epoch()

    def configure_optimizers(self):
        optimizer_name = _cfg_get(self.optimizer_cfg, "name")
        if not optimizer_name:
            raise ValueError("optimizer_cfg.name is required")
        optimizer_cls = getattr(torch.optim, str(optimizer_name), None)
        if optimizer_cls is None:
            raise ValueError(f"unsupported torch optimizer: {optimizer_name}")
        optimizer_params = _config_kwargs(
            _cfg_get(self.optimizer_cfg, "optimizer_params", {})
        )
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("no trainable parameters are available to the optimizer")
        optimizer = optimizer_cls(trainable, **optimizer_params)

        scheduler_name = _cfg_get(self.scheduler_cfg, "name")
        if not scheduler_name:
            return optimizer
        scheduler_cls = getattr(torch.optim.lr_scheduler, str(scheduler_name), None)
        if scheduler_cls is None:
            raise ValueError(f"unsupported torch scheduler: {scheduler_name}")
        scheduler_params = _config_kwargs(
            _cfg_get(self.scheduler_cfg, "scheduler_params", {})
        )
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        scheduler_spec: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": _cfg_get(self.scheduler_cfg, "interval", "step"),
            "frequency": int(_cfg_get(self.scheduler_cfg, "frequency", 1)),
        }
        monitor = _cfg_get(self.scheduler_cfg, "monitor")
        if monitor is not None:
            scheduler_spec["monitor"] = monitor
        return {"optimizer": optimizer, "lr_scheduler": scheduler_spec}


# Compatibility aliases for config experiments and downstream imports.
BaseGenModel = BaseGenerationModel
MusicGenLightningModule = BaseGenerationModel

__all__ = ["BaseGenModel", "BaseGenerationModel", "MusicGenLightningModule"]
