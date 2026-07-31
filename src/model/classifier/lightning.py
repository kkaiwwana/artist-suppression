"""Lightning training wrapper for closed-set artist classification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    value = (
        OmegaConf.to_container(config, resolve=True)
        if isinstance(config, DictConfig)
        else dict(config)
    )
    if not isinstance(value, dict):
        raise TypeError("configuration must resolve to a mapping")
    return value


class ArtistClassifierLightningModule(pl.LightningModule):
    """Train a PyTorch classifier and report top-1/top-5 on every split."""

    def __init__(
        self,
        metrics: Any,
        model_cfg: Any,
        optimizer_cfg: Any,
        scheduler_cfg: Any = None,
        cfg: Any = None,
        log_backend: str = "wandb",
    ) -> None:
        super().__init__()
        del metrics
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.cfg = cfg
        self.log_backend = log_backend
        classifier_cfg = _cfg_get(model_cfg, "classifier", model_cfg)
        self.model = (
            classifier_cfg
            if isinstance(classifier_cfg, nn.Module)
            else instantiate(classifier_cfg)
        )
        if not isinstance(self.model, nn.Module):
            raise TypeError("model_cfg.classifier must instantiate nn.Module")
        names = _cfg_get(model_cfg, "validation_names", ("gt",))
        if names in (None, "auto"):
            names = ("gt",)
        self.validation_names = tuple(str(name) for name in names)
        self.artist_vocabulary: list[str] = []

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        return self.model(
            batch["input_values"],
            batch.get("attention_mask"),
        )

    @staticmethod
    def _accuracies(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        top1 = logits.argmax(dim=-1).eq(labels).float().mean()
        k = min(5, logits.shape[-1])
        top5 = logits.topk(k, dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1)
        return top1, top5.float().mean()

    def _step(self, batch: Mapping[str, Any], prefix: str, *, on_step: bool) -> Tensor:
        labels = torch.as_tensor(batch["labels"], device=self.device).long()
        logits = self(batch)
        loss = F.cross_entropy(logits, labels)
        top1, top5 = self._accuracies(logits.detach(), labels)
        batch_size = int(labels.shape[0])
        self.log_dict(
            {
                f"{prefix}/loss": loss,
                f"{prefix}/top1": top1,
                f"{prefix}/top5": top5,
            },
            on_step=on_step,
            on_epoch=True,
            prog_bar=prefix in {"train", "val/gt"},
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )
        return loss

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "train", on_step=True)

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        del batch_idx
        name = (
            self.validation_names[dataloader_idx]
            if dataloader_idx < len(self.validation_names)
            else f"loader_{dataloader_idx}"
        )
        loss = self._step(batch, f"val/{name}", on_step=False)
        return {"loss": loss.detach()}

    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        del batch_idx, dataloader_idx
        loss = self._step(batch, "test/gt", on_step=False)
        return {"loss": loss.detach()}

    def on_fit_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        vocabulary = getattr(datamodule, "vocabulary", None)
        if vocabulary is not None:
            self.artist_vocabulary = list(vocabulary.index_to_artist)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["artist_vocabulary"] = list(self.artist_vocabulary)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.artist_vocabulary = list(checkpoint.get("artist_vocabulary", ()))

    def configure_optimizers(self):
        optimizer_name = str(_cfg_get(self.optimizer_cfg, "name", "AdamW"))
        optimizer_cls = getattr(torch.optim, optimizer_name, None)
        if optimizer_cls is None:
            raise ValueError(f"unsupported torch optimizer: {optimizer_name}")
        params = _as_dict(_cfg_get(self.optimizer_cfg, "optimizer_params", {}))
        base_lr = float(params.get("lr", 3e-4))
        backbone_lr = _cfg_get(self.optimizer_cfg, "backbone_lr", None)
        backbone_parameters = [
            parameter
            for parameter in self.model.backbone.parameters()
            if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        groups: list[dict[str, Any]] = []
        if head_parameters:
            groups.append({"params": head_parameters, "lr": base_lr})
        if backbone_parameters:
            groups.append(
                {
                    "params": backbone_parameters,
                    "lr": float(backbone_lr if backbone_lr is not None else base_lr),
                }
            )
        if not groups:
            raise RuntimeError("artist classifier has no trainable parameters")
        optimizer = optimizer_cls(groups, **{key: value for key, value in params.items() if key != "lr"})

        scheduler_name = _cfg_get(self.scheduler_cfg, "name")
        if not scheduler_name:
            return optimizer
        scheduler_cls = getattr(torch.optim.lr_scheduler, str(scheduler_name), None)
        if scheduler_cls is None:
            raise ValueError(f"unsupported torch scheduler: {scheduler_name}")
        scheduler = scheduler_cls(
            optimizer,
            **_as_dict(_cfg_get(self.scheduler_cfg, "scheduler_params", {})),
        )
        scheduler_spec: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": _cfg_get(self.scheduler_cfg, "interval", "epoch"),
            "frequency": int(_cfg_get(self.scheduler_cfg, "frequency", 1)),
        }
        monitor = _cfg_get(self.scheduler_cfg, "monitor")
        if monitor is not None:
            scheduler_spec["monitor"] = monitor
        return {"optimizer": optimizer, "lr_scheduler": scheduler_spec}


__all__ = ["ArtistClassifierLightningModule"]
