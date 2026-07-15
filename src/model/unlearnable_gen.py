"""Lightning model that trains MusicGen together with ConceptLearner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from src.model.base import (
    BaseGenerationModel,
    _cfg_get,
    _instantiate_component,
)


class UnlearnableGenerationModel(BaseGenerationModel):
    """Joint MusicGen-adapter and ConceptLearner training module.

    The generator's original language-model loss is computed by ``MusciGen``.
    ConceptLearner is prepared once per batch, installed on decoder-layer
    hidden-state hooks for the generator call, and removed immediately after
    that call.  Its router and style-bank losses are then added to the LM loss.

    Recommended control fields in a batch are:

    - ``concept_text_features``: ``[B, D]`` or ``[B, L, D]`` router input;
    - ``concept_targets``: categorical ``[B]`` or dense multilabel targets;
    - ``suppression_mask``: optional explicit boolean ``[B]`` override.

    During training, suppression rows may instead be sampled from the Hydra
    ``suppression`` configuration.  Retain rows add the routed residual while
    suppression rows subtract it and optimize a bounded reverse-CE margin.

    Generator fields keep the MusicGen contract documented in
    ``src.model.gen.musicgen``.
    """

    CONTROL_BATCH_KEYS = {
        "concept_condition",
        "concept_ids",
        "concept_weights",
        "concept_targets",
        "concept_text_features",
        "concept_text_attention_mask",
        "text_features",
        "text_feature_attention_mask",
        "available_concept_mask",
        "forgotten_concept_ids",
        "concept_temperature",
        "renormalize_concept_weights",
        "sample_concept_style",
        "suppression_mask",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        concept_cfg = _cfg_get(self.model_cfg, "concept_learner")
        self.concept_learner = _instantiate_component(concept_cfg)
        self.loss_cfg = _cfg_get(self.model_cfg, "losses", {})
        self.suppression_cfg = _cfg_get(self.model_cfg, "suppression", {})
        self.hook_blocks = _cfg_get(self.model_cfg, "hook_blocks")
        self.intervention_scale = _cfg_get(
            self.model_cfg,
            "intervention_scale",
            1.0,
        )
        probability = float(_cfg_get(self.suppression_cfg, "probability", 0.0))
        weight = float(_cfg_get(self.suppression_cfg, "weight", 0.0))
        margin = float(_cfg_get(self.suppression_cfg, "margin", 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("suppression.probability must be in [0, 1]")
        if weight < 0.0:
            raise ValueError("suppression.weight must be non-negative")
        if margin < 0.0:
            raise ValueError("suppression.margin must be non-negative")

    @staticmethod
    def _first(batch: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in batch and batch[key] is not None:
                return batch[key]
        return None

    def _prepare_concept_condition(self, batch: Mapping[str, Any]):
        supplied = batch.get("concept_condition")
        if supplied is not None:
            return supplied

        text_features = self._first(
            batch,
            "concept_text_features",
            "text_features",
        )
        concept_weights = batch.get("concept_weights")
        concept_ids = batch.get("concept_ids")
        kwargs: Dict[str, Any] = {}
        if text_features is not None:
            kwargs["text_features"] = text_features
            text_mask = self._first(
                batch,
                "concept_text_attention_mask",
                "text_feature_attention_mask",
            )
            if text_mask is not None:
                kwargs["text_attention_mask"] = text_mask
        elif concept_weights is not None:
            kwargs["concept_weights"] = concept_weights
        elif concept_ids is not None:
            kwargs["concept_ids"] = concept_ids
        else:
            raise ValueError(
                "ConceptLearner training requires one of concept_condition, "
                "concept_text_features/text_features, concept_weights, or concept_ids"
            )

        optional_keys = {
            "available_concept_mask": "available_concept_mask",
            "forgotten_concept_ids": "forgotten_concept_ids",
            "concept_temperature": "temperature",
            "renormalize_concept_weights": "renormalize_weights",
            "sample_concept_style": "sample_style",
            "suppression_mask": "suppression_mask",
        }
        for batch_key, argument_name in optional_keys.items():
            if batch_key in batch and batch[batch_key] is not None:
                kwargs[argument_name] = batch[batch_key]
        return self.concept_learner.prepare_condition(**kwargs)

    def _text_router_batch(self, batch: Mapping[str, Any]) -> Tensor:
        text_features = self._first(
            batch,
            "concept_text_features",
            "text_features",
        )
        if not isinstance(text_features, Tensor) or text_features.ndim < 2:
            raise ValueError(
                "suppression training requires concept_text_features/text_features "
                "with shape [B, D] or [B, L, D]"
            )
        return text_features

    def _ensure_mixed_suppression_mask(self, mask: Tensor) -> Tensor:
        if mask.numel() < 2 or not bool(
            _cfg_get(self.suppression_cfg, "ensure_mixed_batch", True)
        ):
            return mask
        mask = mask.clone()
        if not mask.any():
            mask[torch.randint(mask.numel(), (), device=mask.device)] = True
        elif mask.all():
            mask[torch.randint(mask.numel(), (), device=mask.device)] = False
        return mask

    def _resolve_suppression_mask(
        self,
        batch: Mapping[str, Any],
        *,
        stage: str,
    ) -> Tensor:
        text_features = self._text_router_batch(batch)
        batch_size = text_features.shape[0]
        supplied = batch.get("suppression_mask")
        if supplied is not None:
            mask = torch.as_tensor(supplied, device=text_features.device).bool()
            if mask.ndim == 0:
                mask = mask.expand(batch_size)
            if mask.shape != (batch_size,):
                raise ValueError("suppression_mask must have shape [batch]")
            return self._ensure_mixed_suppression_mask(mask) if stage == "train" else mask

        enabled = bool(_cfg_get(self.suppression_cfg, "enabled", False))
        probability = float(_cfg_get(self.suppression_cfg, "probability", 0.0))
        if stage != "train" or not enabled or probability <= 0.0:
            return torch.zeros(
                batch_size,
                device=text_features.device,
                dtype=torch.bool,
            )
        if batch_size < 2 and bool(
            _cfg_get(self.suppression_cfg, "ensure_mixed_batch", True)
        ):
            return torch.zeros(
                batch_size,
                device=text_features.device,
                dtype=torch.bool,
            )
        mask = torch.rand(batch_size, device=text_features.device).lt(probability)
        return self._ensure_mixed_suppression_mask(mask)

    def _generator_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in batch.items()
            if key not in self.CONTROL_BATCH_KEYS
        }

    def _concept_targets(self, batch: Mapping[str, Any], condition: Any):
        if getattr(condition, "router_logits", None) is None:
            return None
        targets = batch.get("concept_targets")
        if targets is None:
            targets = batch.get("concept_ids")
        return targets

    @staticmethod
    def _margin_suppression_losses(
        sample_losses: Tensor,
        suppression_mask: Tensor,
        margin: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Split retain CE and bounded reverse CE for suppression rows."""

        retain_mask = ~suppression_mask
        if not retain_mask.any():
            raise ValueError(
                "margin suppression requires at least one retain sample per batch"
            )
        retain_loss = sample_losses[retain_mask].mean()
        zero = sample_losses.sum() * 0.0
        if not suppression_mask.any():
            return retain_loss, zero, zero

        suppressed_sample_losses = sample_losses[suppression_mask]
        suppressed_raw_loss = suppressed_sample_losses.mean()
        margin_target = retain_loss.detach() + margin
        suppression_margin_loss = torch.relu(
            margin_target - suppressed_sample_losses
        ).mean()
        return retain_loss, suppressed_raw_loss, suppression_margin_loss

    def _run_generator_with_control(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        *,
        stage: str,
    ) -> Tuple[Dict[str, Any], Any]:
        condition = self._prepare_concept_condition(batch)
        handles = self.model.register_concept_learner_hooks(
            self.concept_learner,
            condition=condition,
            module_names=self.hook_blocks,
            intervention_scale=self.intervention_scale,
        )
        try:
            generator_batch = self._generator_batch(batch)
            result = (
                self.model.training_step(generator_batch, batch_idx)
                if stage == "train"
                else self.model.validation_step(generator_batch, batch_idx)
            )
        finally:
            self.model.remove_hidden_state_hooks(handles)
        return result, condition

    def _controlled_shared_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        *,
        stage: str,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        controlled_batch = dict(batch)
        suppression_mask = self._resolve_suppression_mask(batch, stage=stage)
        controlled_batch["suppression_mask"] = suppression_mask
        result, condition = self._run_generator_with_control(
            controlled_batch,
            batch_idx,
            stage=stage,
        )
        concept_targets = self._concept_targets(controlled_batch, condition)
        if concept_targets is None:
            raise ValueError(
                "text-router training requires concept_targets (or categorical "
                "concept_ids as a target alias)"
            )
        concept_losses = self.concept_learner.compute_losses(
            condition=condition,
            concept_targets=concept_targets,
            routing_loss_type=_cfg_get(self.loss_cfg, "routing_loss_type"),
            router_weight=float(_cfg_get(self.loss_cfg, "router_weight", 1.0)),
            style_bank_l2_weight=float(
                _cfg_get(self.loss_cfg, "style_bank_l2_weight", 0.0)
            ),
            style_bank_kl_weight=float(
                _cfg_get(self.loss_cfg, "style_bank_kl_weight", 0.0)
            ),
        )
        sample_losses = result.get("sample_losses")
        if not isinstance(sample_losses, Tensor) or sample_losses.ndim != 1:
            raise RuntimeError(
                "the generator must return sample_losses with shape [batch]"
            )
        if sample_losses.shape[0] != suppression_mask.shape[0]:
            raise RuntimeError(
                "generator sample_losses and suppression_mask batch sizes differ"
            )
        margin = float(_cfg_get(self.suppression_cfg, "margin", 0.0))
        (
            retain_loss,
            suppressed_raw_loss,
            suppression_margin_loss,
        ) = self._margin_suppression_losses(
            sample_losses,
            suppression_mask,
            margin,
        )

        suppression_weight = float(
            _cfg_get(self.suppression_cfg, "weight", 0.0)
        )
        generator_objective = (
            retain_loss + suppression_weight * suppression_margin_loss
        )
        generator_weight = float(
            _cfg_get(self.loss_cfg, "generator_loss_weight", 1.0)
        )
        weighted_generator_loss = generator_objective * generator_weight
        total_loss = weighted_generator_loss + concept_losses["loss"]

        loss_dict: Dict[str, Tensor] = {
            "musicgen_lm": result["loss"],
            "musicgen_retain": retain_loss,
            "musicgen_suppressed_raw": suppressed_raw_loss,
            "suppression_margin": suppression_margin_loss,
            "suppression_rate": suppression_mask.float().mean(),
            "musicgen_objective": generator_objective,
            "musicgen_lm_weighted": weighted_generator_loss,
            "concept_auxiliary": concept_losses["loss"],
        }
        loss_dict.update(
            {
                f"concept_{name}": value
                for name, value in concept_losses.items()
                if name != "loss"
            }
        )
        self._log_losses(total_loss, loss_dict, stage=stage)
        return total_loss, loss_dict

    def forward(self, batch: Mapping[str, Any], **kwargs):
        condition = self._prepare_concept_condition(batch)
        handles = self.model.register_concept_learner_hooks(
            self.concept_learner,
            condition=condition,
            module_names=self.hook_blocks,
            intervention_scale=self.intervention_scale,
        )
        try:
            return self.model(self._generator_batch(batch), **kwargs)
        finally:
            self.model.remove_hidden_state_hooks(handles)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        total_loss, _ = self._controlled_shared_step(
            batch,
            batch_idx,
            stage="train",
        )
        return total_loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Tensor]:
        del dataloader_idx
        total_loss, loss_dict = self._controlled_shared_step(
            batch,
            batch_idx,
            stage="val",
        )
        return {
            "loss": total_loss.detach(),
            **{name: value.detach() for name, value in loss_dict.items()},
        }

    @torch.no_grad()
    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        """Generate audio while ConceptLearner intervenes in decoder blocks."""

        del dataloader_idx
        condition = self._prepare_concept_condition(batch)
        handles = self.model.register_concept_learner_hooks(
            self.concept_learner,
            condition=condition,
            module_names=self.hook_blocks,
            intervention_scale=self.intervention_scale,
        )
        try:
            audio_values, prompts = self._generate_test_batch(
                self._generator_batch(batch)
            )
        finally:
            self.model.remove_hidden_state_hooks(handles)

        output = self._test_output(audio_values, prompts, batch_idx)
        output["concept_weights"] = condition.weights.detach()
        output["concept_active"] = condition.active.detach()
        if condition.suppression_mask is not None:
            output["suppression_mask"] = condition.suppression_mask.detach()
        return output


UnlearnableGenModel = UnlearnableGenerationModel
UnlearnableMusicGenLightningModule = UnlearnableGenerationModel

__all__ = [
    "UnlearnableGenModel",
    "UnlearnableGenerationModel",
    "UnlearnableMusicGenLightningModule",
]
