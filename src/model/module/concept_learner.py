"""Granularity-agnostic concept control for an external generative model.

One :class:`ConceptLearner` instance represents exactly one experimental
setup, for example artists *or* genres.  The two granularities intentionally
do not coexist in a learner: changing granularity means constructing another
instance with a different concept vocabulary and routing mode.

The module is independent of the concrete music generator.  A caller first
builds one :class:`ConceptCondition` from text features, labels, or explicit
mixture weights, then reuses it in hooks attached to different transformer
blocks.  Every block index has its own learned depth embedding and
intervention strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ConceptIds = Union[int, Sequence[int], Tensor]


@dataclass
class ConceptCondition:
    """A generator-independent control condition for one concept setup.

    Attributes:
        style: Synthesized control tensor of shape ``[batch, style_dim]``.
        weights: Concept mixture weights of shape ``[batch, num_concepts]``.
        active: Binary per-sample flag.  It is zero when every concept has
            been masked and prevents an additive decoder bias from leaking
            into the no-concept case.
        router_logits: Unmasked router logits used for supervision.  It is
            ``None`` for teacher-forced ids and manually supplied weights.
        availability_mask: Final mask applied before style-bank synthesis.
    """

    style: Tensor
    weights: Tensor
    active: Tensor
    router_logits: Optional[Tensor] = None
    availability_mask: Optional[Tensor] = None


class ConceptStyleBank(nn.Module):
    """Learnable fixed or diagonal-Gaussian concept representations."""

    SUPPORTED_REPRESENTATIONS = {"fixed", "gaussian"}

    def __init__(
        self,
        num_concepts: int,
        style_dim: int,
        representation: str = "fixed",
        init_std: float = 0.02,
        initial_log_std: float = -2.0,
    ) -> None:
        super().__init__()
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        if style_dim <= 0:
            raise ValueError("style_dim must be positive")
        if representation not in self.SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {self.SUPPORTED_REPRESENTATIONS}, "
                f"got {representation!r}"
            )

        self.num_concepts = num_concepts
        self.style_dim = style_dim
        self.representation = representation

        if representation == "fixed":
            self.embeddings = nn.Embedding(num_concepts, style_dim)
            nn.init.normal_(self.embeddings.weight, std=init_std)
            self.register_parameter("means", None)
            self.register_parameter("log_stds", None)
        else:
            self.register_module("embeddings", None)
            self.means = nn.Parameter(torch.empty(num_concepts, style_dim))
            self.log_stds = nn.Parameter(
                torch.full((num_concepts, style_dim), initial_log_std)
            )
            nn.init.normal_(self.means, std=init_std)

    @property
    def deterministic_embeddings(self) -> Tensor:
        """Return fixed embeddings or Gaussian means for inspection."""

        if self.representation == "fixed":
            return self.embeddings.weight
        return self.means

    def forward(self, weights: Tensor, sample: Optional[bool] = None) -> Tensor:
        """Synthesize a control from concept mixture weights.

        For Gaussian concepts, the mixture is treated as a weighted sum of
        independent diagonal Gaussians.  Its variance is therefore
        ``sum_j weights_j**2 * variance_j``.
        """

        if weights.ndim != 2 or weights.shape[-1] != self.num_concepts:
            raise ValueError(
                f"weights must have shape [batch, {self.num_concepts}], "
                f"got {tuple(weights.shape)}"
            )
        weights = weights.to(
            device=self.deterministic_embeddings.device,
            dtype=self.deterministic_embeddings.dtype,
        )

        if self.representation == "fixed":
            return weights @ self.embeddings.weight

        mean = weights @ self.means
        should_sample = self.training if sample is None else sample
        if not should_sample:
            return mean

        variances = torch.exp(2.0 * self.log_stds)
        mixed_variance = weights.square() @ variances
        mixed_std = torch.sqrt(mixed_variance.clamp_min(1e-12))
        # An all-masked row must remain exactly zero, including sampling noise.
        mixed_std = mixed_std * (mixed_variance > 0).to(mixed_std)
        return mean + torch.randn_like(mean) * mixed_std

    def regularization_losses(self) -> Dict[str, Tensor]:
        """Return unweighted style-bank regularizers."""

        embeddings = self.deterministic_embeddings
        losses = {"style_bank_l2": embeddings.square().mean()}
        if self.representation == "gaussian":
            variance = torch.exp(2.0 * self.log_stds)
            losses["style_bank_kl"] = 0.5 * (
                self.means.square() + variance - 1.0 - 2.0 * self.log_stds
            ).mean()
        return losses


class _DepthConditionedStyleDecoder(nn.Module):
    """Decode a concept control into a hidden-state residual."""

    def __init__(
        self,
        hidden_dim: int,
        style_dim: int,
        depth_dim: int,
        controller_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, controller_dim)
        self.condition_projection = nn.Linear(style_dim + depth_dim, controller_dim)
        self.output_projection = nn.Linear(controller_dim, hidden_dim)

    def forward(self, hidden_state: Tensor, style: Tensor, depth: Tensor) -> Tensor:
        hidden_features = self.hidden_projection(self.hidden_norm(hidden_state))
        condition = self.condition_projection(torch.cat((style, depth), dim=-1))
        broadcast_shape = (condition.shape[0],) + (1,) * (hidden_state.ndim - 2) + (
            condition.shape[-1],
        )
        return self.output_projection(
            F.gelu(hidden_features + condition.view(broadcast_shape))
        )


class _DepthConditionedStylePredictor(nn.Module):
    """Estimate the existing hidden-state style residual for replacement."""

    def __init__(self, hidden_dim: int, depth_dim: int, controller_dim: int) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, controller_dim)
        self.depth_projection = nn.Linear(depth_dim, controller_dim)
        self.output_projection = nn.Linear(controller_dim, hidden_dim)

    def forward(self, hidden_state: Tensor, depth: Tensor) -> Tensor:
        hidden_features = self.hidden_projection(self.hidden_norm(hidden_state))
        depth_features = self.depth_projection(depth)
        broadcast_shape = (depth.shape[0],) + (1,) * (hidden_state.ndim - 2) + (
            depth_features.shape[-1],
        )
        return self.output_projection(
            F.gelu(hidden_features + depth_features.view(broadcast_shape))
        )


class ConceptLearner(nn.Module):
    """External controller for one concept granularity.

    Args:
        concept_name: Human-readable setup name such as ``"artist"`` or
            ``"genre"``.  It is metadata only and does not introduce semantic
            branches into the implementation.
        routing_mode: ``"categorical"`` uses softmax/CE and suits mutually
            exclusive labels.  ``"multilabel"`` uses sigmoid/BCE and suits
            samples with several simultaneous concepts.
        normalize_multilabel_weights: Normalize sigmoid/multi-hot weights to
            sum to one before style synthesis.  This prevents samples with
            more labels from receiving a larger intervention merely because
            of their label count.

    A normal hook workflow prepares the condition once and reuses it::

        condition = learner.prepare_condition(
            text_features=pooled_text_features,
            forgotten_concept_ids=[concept_to_suppress],
        )
        changed_hidden = learner(hidden, block_index=i, condition=condition)

    The style bank and controller learn from the outer generator loss.
    :meth:`compute_losses` supplies router supervision and optional style-bank
    regularization without depending on a concrete generator.
    """

    SUPPORTED_INTERVENTIONS = {"additive", "replacement"}
    SUPPORTED_ROUTING_MODES = {"categorical", "multilabel"}

    def __init__(
        self,
        num_concepts: int,
        hidden_dim: int,
        style_dim: int,
        num_blocks: int,
        text_feature_dim: Optional[int] = None,
        *,
        concept_name: str = "concept",
        routing_mode: str = "categorical",
        normalize_multilabel_weights: bool = True,
        representation: str = "fixed",
        intervention: str = "additive",
        controller_dim: Optional[int] = None,
        depth_dim: int = 32,
        router_hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        initial_block_scale: float = 0.1,
        intervention_blocks: Optional[Sequence[int]] = None,
        sample_gaussian_during_training: bool = True,
    ) -> None:
        super().__init__()
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        if hidden_dim <= 0 or style_dim <= 0 or num_blocks <= 0 or depth_dim <= 0:
            raise ValueError(
                "hidden_dim, style_dim, num_blocks and depth_dim must be positive"
            )
        if not concept_name.strip():
            raise ValueError("concept_name must not be empty")
        if routing_mode not in self.SUPPORTED_ROUTING_MODES:
            raise ValueError(
                f"routing_mode must be one of {self.SUPPORTED_ROUTING_MODES}, "
                f"got {routing_mode!r}"
            )
        if intervention not in self.SUPPORTED_INTERVENTIONS:
            raise ValueError(
                f"intervention must be one of {self.SUPPORTED_INTERVENTIONS}, "
                f"got {intervention!r}"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.num_blocks = num_blocks
        self.text_feature_dim = text_feature_dim
        self.concept_name = concept_name
        self.routing_mode = routing_mode
        self.normalize_multilabel_weights = normalize_multilabel_weights
        self.intervention = intervention
        self.temperature = temperature
        self.sample_gaussian_during_training = sample_gaussian_during_training

        controller_dim = controller_dim or hidden_dim
        if controller_dim <= 0:
            raise ValueError("controller_dim must be positive")

        self.style_bank = ConceptStyleBank(
            num_concepts=num_concepts,
            style_dim=style_dim,
            representation=representation,
        )

        if text_feature_dim is None:
            self.router = None
        else:
            if text_feature_dim <= 0:
                raise ValueError("text_feature_dim must be positive when supplied")
            router_hidden_dim = router_hidden_dim or max(style_dim, text_feature_dim)
            if router_hidden_dim <= 0:
                raise ValueError("router_hidden_dim must be positive")
            self.router = nn.Sequential(
                nn.LayerNorm(text_feature_dim),
                nn.Linear(text_feature_dim, router_hidden_dim),
                nn.GELU(),
                nn.Linear(router_hidden_dim, num_concepts),
            )

        self.depth_embeddings = nn.Embedding(num_blocks, depth_dim)
        nn.init.normal_(self.depth_embeddings.weight, std=0.02)
        self.block_scales = nn.Parameter(
            torch.full((num_blocks,), float(initial_block_scale))
        )

        enabled_blocks = torch.zeros(num_blocks, dtype=torch.bool)
        if intervention_blocks is None:
            enabled_blocks.fill_(True)
        else:
            for block_index in intervention_blocks:
                self._validate_block_index(block_index)
                enabled_blocks[block_index] = True
        self.register_buffer("intervention_mask", enabled_blocks)

        # Runtime suppression state is deliberately not checkpointed.
        self.register_buffer(
            "concept_availability",
            torch.ones(num_concepts, dtype=torch.bool),
            persistent=False,
        )

        self.style_decoder = _DepthConditionedStyleDecoder(
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            depth_dim=depth_dim,
            controller_dim=controller_dim,
        )
        self.style_predictor = (
            _DepthConditionedStylePredictor(hidden_dim, depth_dim, controller_dim)
            if intervention == "replacement"
            else None
        )

    def extra_repr(self) -> str:
        return (
            f"concept_name={self.concept_name!r}, num_concepts={self.num_concepts}, "
            f"routing_mode={self.routing_mode!r}, intervention={self.intervention!r}"
        )

    def _validate_block_index(self, block_index: int) -> None:
        if not isinstance(block_index, int):
            raise TypeError("block_index must be a Python int")
        if not 0 <= block_index < self.num_blocks:
            raise IndexError(
                f"block_index must be in [0, {self.num_blocks}), got {block_index}"
            )

    def _normalise_global_concept_ids(self, concept_ids: ConceptIds) -> Tensor:
        ids = torch.as_tensor(concept_ids, device=self.concept_availability.device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("global concept ids must be scalar or one-dimensional")
        ids = ids.to(dtype=torch.long)
        if ids.numel() and ((ids < 0).any() or (ids >= self.num_concepts).any()):
            raise IndexError(f"concept ids must be in [0, {self.num_concepts})")
        return ids

    @torch.no_grad()
    def forget_concepts(self, concept_ids: ConceptIds) -> None:
        """Make concepts unavailable for subsequent condition preparation."""

        ids = self._normalise_global_concept_ids(concept_ids)
        self.concept_availability[ids] = False

    @torch.no_grad()
    def restore_concepts(self, concept_ids: Optional[ConceptIds] = None) -> None:
        """Restore selected concepts, or all concepts when ids are omitted."""

        if concept_ids is None:
            self.concept_availability.fill_(True)
            return
        ids = self._normalise_global_concept_ids(concept_ids)
        self.concept_availability[ids] = True

    @torch.no_grad()
    def set_concept_availability(self, available_mask: Tensor) -> None:
        """Replace the runtime concept-availability mask."""

        mask = torch.as_tensor(available_mask, device=self.concept_availability.device)
        if mask.shape != (self.num_concepts,):
            raise ValueError(
                f"available_mask must have shape [{self.num_concepts}], "
                f"got {tuple(mask.shape)}"
            )
        self.concept_availability.copy_(mask.bool())

    def _pool_text_features(
        self,
        text_features: Tensor,
        text_attention_mask: Optional[Tensor],
    ) -> Tensor:
        if text_features.ndim == 2:
            return text_features
        if text_features.ndim != 3:
            raise ValueError(
                "text_features must have shape [batch, dim] or "
                "[batch, sequence, dim]"
            )
        if text_attention_mask is None:
            return text_features.mean(dim=1)
        if text_attention_mask.shape != text_features.shape[:2]:
            raise ValueError("text_attention_mask must match [batch, sequence]")
        mask = text_attention_mask.to(text_features).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (text_features * mask).sum(dim=1) / denominator

    def route(
        self,
        text_features: Tensor,
        text_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return unmasked router logits for prediction and supervision."""

        if self.router is None:
            raise RuntimeError(
                "This ConceptLearner has no router; set text_feature_dim in __init__"
            )
        pooled_features = self._pool_text_features(
            text_features,
            text_attention_mask=text_attention_mask,
        )
        if pooled_features.shape[-1] != self.text_feature_dim:
            raise ValueError(
                f"expected text feature dimension {self.text_feature_dim}, "
                f"got {pooled_features.shape[-1]}"
            )
        return self.router(pooled_features)

    def _build_availability_mask(
        self,
        batch_size: int,
        device: torch.device,
        available_concept_mask: Optional[Tensor],
        forgotten_concept_ids: Optional[ConceptIds],
    ) -> Tensor:
        mask = self.concept_availability.to(device=device).unsqueeze(0).expand(
            batch_size, -1
        )
        mask = mask.clone()

        if available_concept_mask is not None:
            supplied_mask = torch.as_tensor(
                available_concept_mask,
                device=device,
            ).bool()
            if supplied_mask.ndim == 1:
                supplied_mask = supplied_mask.unsqueeze(0)
            if supplied_mask.shape not in {
                (1, self.num_concepts),
                (batch_size, self.num_concepts),
            }:
                raise ValueError(
                    "available_concept_mask must have shape [num_concepts] or "
                    "[batch, num_concepts]"
                )
            mask &= supplied_mask.expand(batch_size, -1)

        if forgotten_concept_ids is None:
            return mask

        forgotten = torch.as_tensor(forgotten_concept_ids, device=device)
        if forgotten.ndim == 0:
            forgotten = forgotten.view(1)
        forgotten = forgotten.to(dtype=torch.long)

        if forgotten.ndim == 1:
            valid = forgotten >= 0
            selected = forgotten[valid]
            if selected.numel() and (selected >= self.num_concepts).any():
                raise IndexError(f"concept ids must be in [0, {self.num_concepts})")
            mask[:, selected] = False
        elif forgotten.ndim == 2:
            if forgotten.shape[0] != batch_size:
                raise ValueError(
                    "per-sample forgotten ids must have shape [batch, count]"
                )
            valid = forgotten >= 0  # -1 is padding.
            if valid.any() and (forgotten[valid] >= self.num_concepts).any():
                raise IndexError(f"concept ids must be in [0, {self.num_concepts})")
            rows = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(
                forgotten
            )
            mask[rows[valid], forgotten[valid]] = False
        else:
            raise ValueError(
                "forgotten_concept_ids must be scalar, [count], or [batch, count]"
            )
        return mask

    @staticmethod
    def _normalise_weights(weights: Tensor) -> Tensor:
        total = weights.sum(dim=-1, keepdim=True)
        return torch.where(
            total > 0,
            weights / total.clamp_min(torch.finfo(weights.dtype).tiny),
            torch.zeros_like(weights),
        )

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor, temperature: float) -> Tensor:
        """Softmax that returns zero rather than NaN for all-masked rows."""

        masked_logits = (logits / temperature).masked_fill(~mask, -torch.inf)
        row_max = masked_logits.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        unnormalised = torch.exp(masked_logits - row_max)
        return ConceptLearner._normalise_weights(unnormalised)

    def _router_weights(
        self,
        logits: Tensor,
        availability: Tensor,
        temperature: float,
        renormalize_weights: bool,
    ) -> Tensor:
        if self.routing_mode == "categorical":
            return self._masked_softmax(logits, availability, temperature)

        masked_logits = (logits / temperature).masked_fill(
            ~availability,
            -torch.inf,
        )
        weights = torch.sigmoid(masked_logits)
        if renormalize_weights:
            weights = self._normalise_weights(weights)
        return weights

    def _concept_ids_to_weights(
        self,
        concept_ids: ConceptIds,
    ) -> Tensor:
        """Convert categorical or padded multilabel ids to a dense matrix.

        A one-dimensional tensor represents one id per batch item.  For a
        multilabel sample, use ``[batch, max_labels]`` and pad with ``-1``.
        """

        ids = torch.as_tensor(concept_ids)
        if ids.ndim == 0:
            ids = ids.view(1)
        if ids.ndim == 1:
            ids = ids.unsqueeze(1)
        elif ids.ndim == 2 and self.routing_mode != "multilabel":
            raise ValueError(
                "two-dimensional concept_ids are only valid in multilabel mode"
            )
        elif ids.ndim != 2:
            raise ValueError(
                "concept_ids must be scalar, [batch], or [batch, max_labels]"
            )

        ids = ids.to(dtype=torch.long)
        valid = ids >= 0
        if valid.any() and (ids[valid] >= self.num_concepts).any():
            raise IndexError(f"concept ids must be in [0, {self.num_concepts})")
        if self.routing_mode == "categorical" and not valid.all():
            raise ValueError("categorical concept_ids cannot contain padding")

        weights = torch.zeros(
            ids.shape[0],
            self.num_concepts,
            device=ids.device,
            dtype=self.style_bank.deterministic_embeddings.dtype,
        )
        rows = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1).expand_as(ids)
        weights[rows[valid], ids[valid]] = 1.0
        return weights

    def prepare_condition(
        self,
        *,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        concept_ids: Optional[ConceptIds] = None,
        concept_weights: Optional[Tensor] = None,
        available_concept_mask: Optional[Tensor] = None,
        forgotten_concept_ids: Optional[ConceptIds] = None,
        temperature: Optional[float] = None,
        renormalize_weights: Optional[bool] = None,
        sample_style: Optional[bool] = None,
    ) -> ConceptCondition:
        """Route or teacher-force concepts and synthesize a condition.

        Exactly one of ``text_features``, ``concept_ids`` and
        ``concept_weights`` must be supplied.  Availability masking always
        happens before style-bank synthesis.
        """

        sources = (
            text_features is not None,
            concept_ids is not None,
            concept_weights is not None,
        )
        if sum(sources) != 1:
            raise ValueError(
                "supply exactly one of text_features, concept_ids, or concept_weights"
            )

        if renormalize_weights is None:
            renormalize_weights = (
                True
                if self.routing_mode == "categorical"
                else self.normalize_multilabel_weights
            )

        router_logits: Optional[Tensor] = None
        raw_weights: Optional[Tensor] = None
        if text_features is not None:
            router_logits = self.route(text_features, text_attention_mask)
            batch_size = router_logits.shape[0]
            device = router_logits.device
        elif concept_weights is not None:
            raw_weights = torch.as_tensor(concept_weights)
            if raw_weights.ndim == 1:
                raw_weights = raw_weights.unsqueeze(0)
            if raw_weights.ndim != 2 or raw_weights.shape[-1] != self.num_concepts:
                raise ValueError(
                    f"concept_weights must have shape [batch, {self.num_concepts}]"
                )
            if (raw_weights < 0).any():
                raise ValueError("concept_weights must be non-negative")
            raw_weights = raw_weights.to(
                dtype=self.style_bank.deterministic_embeddings.dtype
            )
            batch_size = raw_weights.shape[0]
            device = raw_weights.device
        else:
            raw_weights = self._concept_ids_to_weights(concept_ids)
            batch_size = raw_weights.shape[0]
            device = raw_weights.device

        availability = self._build_availability_mask(
            batch_size=batch_size,
            device=device,
            available_concept_mask=available_concept_mask,
            forgotten_concept_ids=forgotten_concept_ids,
        )

        if router_logits is not None:
            effective_temperature = self.temperature if temperature is None else temperature
            if effective_temperature <= 0:
                raise ValueError("temperature must be positive")
            weights = self._router_weights(
                router_logits,
                availability,
                effective_temperature,
                renormalize_weights,
            )
        else:
            weights = raw_weights.to(device=device)
            weights = weights * availability.to(dtype=weights.dtype)
            if renormalize_weights:
                weights = self._normalise_weights(weights)

        weights = weights.to(
            device=self.style_bank.deterministic_embeddings.device,
            dtype=self.style_bank.deterministic_embeddings.dtype,
        )
        availability = availability.to(device=weights.device)
        active = (weights.sum(dim=-1) > 0).to(dtype=weights.dtype)

        if sample_style is None and self.style_bank.representation == "gaussian":
            sample_style = self.training and self.sample_gaussian_during_training
        style = self.style_bank(weights, sample=sample_style)
        return ConceptCondition(
            style=style,
            weights=weights,
            active=active,
            router_logits=router_logits,
            availability_mask=availability,
        )

    @staticmethod
    def _batch_scale(scale: Union[float, Tensor], hidden_state: Tensor) -> Tensor:
        scale_tensor = torch.as_tensor(
            scale,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        if scale_tensor.ndim == 0:
            return scale_tensor
        if scale_tensor.ndim != 1 or scale_tensor.shape[0] != hidden_state.shape[0]:
            raise ValueError("intervention_scale must be scalar or have shape [batch]")
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        return scale_tensor.view(shape)

    def forward(
        self,
        hidden_state: Tensor,
        block_index: int,
        condition: Optional[ConceptCondition] = None,
        *,
        style_condition: Optional[Tensor] = None,
        intervention_scale: Union[float, Tensor] = 1.0,
        return_details: bool = False,
        **condition_kwargs,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Apply a depth-conditioned intervention to a captured hidden state.

        ``hidden_state`` may contain arbitrary token/spatial axes between its
        batch and hidden dimensions.  A prepared condition is preferred when
        several hooks use the same text prompt.
        """

        self._validate_block_index(block_index)
        if hidden_state.ndim < 2 or hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden_state must end in hidden_dim={self.hidden_dim}, "
                f"got {tuple(hidden_state.shape)}"
            )
        if condition is not None and (style_condition is not None or condition_kwargs):
            raise ValueError(
                "condition is mutually exclusive with style_condition and "
                "condition-building arguments"
            )
        if style_condition is not None and condition_kwargs:
            raise ValueError(
                "style_condition is mutually exclusive with condition-building arguments"
            )

        if condition is None:
            if style_condition is None:
                condition = self.prepare_condition(**condition_kwargs)
            else:
                style = torch.as_tensor(style_condition)
                if style.ndim == 1:
                    style = style.unsqueeze(0)
                if style.ndim != 2 or style.shape[-1] != self.style_dim:
                    raise ValueError(
                        f"style_condition must have shape [batch, {self.style_dim}]"
                    )
                condition = ConceptCondition(
                    style=style,
                    weights=torch.empty(
                        style.shape[0],
                        0,
                        device=style.device,
                        dtype=style.dtype,
                    ),
                    active=torch.ones(
                        style.shape[0],
                        device=style.device,
                        dtype=style.dtype,
                    ),
                )

        style = condition.style
        active = condition.active
        if style.shape[0] == 1 and hidden_state.shape[0] != 1:
            style = style.expand(hidden_state.shape[0], -1)
            active = active.expand(hidden_state.shape[0])
        elif style.shape[0] != hidden_state.shape[0]:
            raise ValueError("condition batch size must match hidden_state batch size")

        style = style.to(device=hidden_state.device, dtype=hidden_state.dtype)
        active = active.to(device=hidden_state.device, dtype=hidden_state.dtype)
        depth = self.depth_embeddings.weight[block_index].to(
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        depth = depth.unsqueeze(0).expand(hidden_state.shape[0], -1)

        if not bool(self.intervention_mask[block_index]):
            zero_residual = torch.zeros_like(hidden_state)
            if return_details:
                return {
                    "hidden_state": hidden_state,
                    "style_residual": zero_residual,
                    "predicted_style_residual": zero_residual,
                    "effective_scale": hidden_state.new_zeros(()),
                }
            return hidden_state

        style_residual = self.style_decoder(hidden_state, style, depth)
        active_shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        active = active.view(active_shape)
        if self.style_predictor is None:
            predicted_style_residual = torch.zeros_like(style_residual)
            intervention_residual = active * style_residual
        else:
            predicted_style_residual = self.style_predictor(hidden_state, depth)
            # With no available concept, replacement still removes the
            # estimated original style; only the new style branch is gated.
            intervention_residual = active * style_residual - predicted_style_residual

        layer_scale = self.block_scales[block_index].to(dtype=hidden_state.dtype)
        external_scale = self._batch_scale(intervention_scale, hidden_state)
        effective_scale = layer_scale * external_scale
        changed_hidden_state = hidden_state + effective_scale * intervention_residual

        if return_details:
            return {
                "hidden_state": changed_hidden_state,
                "style_residual": style_residual,
                "predicted_style_residual": predicted_style_residual,
                "effective_scale": effective_scale,
            }
        return changed_hidden_state

    def routing_loss(
        self,
        condition: ConceptCondition,
        concept_targets: Tensor,
        loss_type: Optional[str] = None,
    ) -> Tensor:
        """Supervise router logits with CE, soft CE, or BCE-with-logits.

        When ``loss_type`` is omitted, categorical routing selects CE and
        multilabel routing selects BCE.  Multilabel targets must be a dense
        multi-hot tensor with shape ``[batch, num_concepts]``.
        """

        if condition.router_logits is None:
            raise ValueError("routing loss requires a text-routed ConceptCondition")
        logits = condition.router_logits
        targets = concept_targets.to(device=logits.device)
        if loss_type is None or loss_type == "auto":
            loss_type = "ce" if self.routing_mode == "categorical" else "bce"

        if loss_type == "ce":
            if targets.ndim != 1:
                raise ValueError("CE targets must have shape [batch]")
            return F.cross_entropy(logits, targets.long())
        if loss_type == "soft_ce":
            if targets.shape != logits.shape:
                raise ValueError("soft CE targets must match router logits")
            return -(targets.to(logits) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        if loss_type == "bce":
            if targets.shape != logits.shape:
                raise ValueError(
                    "BCE targets must have shape [batch, num_concepts]"
                )
            return F.binary_cross_entropy_with_logits(logits, targets.to(logits))
        raise ValueError("loss_type must be one of: auto, ce, soft_ce, bce")

    def compute_losses(
        self,
        condition: Optional[ConceptCondition] = None,
        concept_targets: Optional[Tensor] = None,
        *,
        routing_loss_type: Optional[str] = None,
        router_weight: float = 1.0,
        style_bank_l2_weight: float = 0.0,
        style_bank_kl_weight: float = 0.0,
    ) -> Dict[str, Tensor]:
        """Return weighted auxiliary loss and unweighted components."""

        regularizers = self.style_bank.regularization_losses()
        total = regularizers["style_bank_l2"] * style_bank_l2_weight
        losses: Dict[str, Tensor] = {
            "style_bank_l2": regularizers["style_bank_l2"]
        }

        if "style_bank_kl" in regularizers:
            losses["style_bank_kl"] = regularizers["style_bank_kl"]
            total = total + regularizers["style_bank_kl"] * style_bank_kl_weight

        if concept_targets is not None:
            if condition is None:
                raise ValueError("condition is required when concept_targets are supplied")
            router_loss = self.routing_loss(
                condition,
                concept_targets,
                loss_type=routing_loss_type,
            )
            losses["router"] = router_loss
            total = total + router_loss * router_weight

        losses["loss"] = total
        return losses


__all__ = ["ConceptCondition", "ConceptLearner", "ConceptStyleBank"]
