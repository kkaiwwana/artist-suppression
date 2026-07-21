"""Explicit, condition-only concept directions for MusicGen.

The controller deliberately has no router and never reads decoder hidden-state
values.  A caller selects a known concept and one of three directions:

``0``
    Default MusicGen generation. No intervention is applied.
``+1``
    Add the concept-specific direction while learning what the concept means.
``-1``
    Subtract that same direction for an explicit copyright suppression request.

Concept directions are centered against artists that share coarse genres.  A
direction is decoded as ``decoder(artist) - decoder(genre peers)`` so shared
musical structure and decoder biases cancel before it reaches MusicGen.
Per-block gates are parameterized through a bounded ``tanh`` map so training
cannot grow their effective magnitude without limit. Before injection, a
direction whose RMS exceeds the incoming hidden-state RMS is clipped down.
Smaller directions are left untouched, so an immature or nearly-zero residual
is never amplified merely because RMS protection is enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ConceptIds = Union[int, Sequence[int], Tensor]


@dataclass
class ConceptCondition:
    """A fixed explicit control prepared before autoregressive generation.

    Attributes:
        weights: Selected concept mixture, ``[batch, num_concepts]``.
        peer_weights: Genre-matched reference mixture with the selected
            concepts excluded, ``[batch, num_concepts]``.
        direction: Per-sample signed strength: zero, positive, or negative.
        active: One when a non-empty concept selection is present.
    """

    weights: Tensor
    peer_weights: Tensor
    direction: Tensor
    active: Tensor


class ConceptStyleBank(nn.Module):
    """Learn one deterministic embedding for every explicit concept."""

    def __init__(
        self,
        num_concepts: int,
        style_dim: int,
        *,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_concepts <= 0 or style_dim <= 0:
            raise ValueError("num_concepts and style_dim must be positive")
        self.num_concepts = int(num_concepts)
        self.style_dim = int(style_dim)
        self.embeddings = nn.Embedding(num_concepts, style_dim)
        nn.init.normal_(self.embeddings.weight, std=init_std)

    @property
    def deterministic_embeddings(self) -> Tensor:
        return self.embeddings.weight

    def forward(self, weights: Tensor) -> Tensor:
        if weights.ndim != 2 or weights.shape[-1] != self.num_concepts:
            raise ValueError(
                f"weights must have shape [batch, {self.num_concepts}], "
                f"got {tuple(weights.shape)}"
            )
        weights = weights.to(
            device=self.embeddings.weight.device,
            dtype=self.embeddings.weight.dtype,
        )
        return weights @ self.embeddings.weight


class _ConditionOnlyStyleDecoder(nn.Module):
    """Map style and block depth to a token-independent residual."""

    def __init__(
        self,
        hidden_dim: int,
        style_dim: int,
        depth_dim: int,
        controller_dim: int,
    ) -> None:
        super().__init__()
        self.condition_projection = nn.Linear(
            style_dim + depth_dim,
            controller_dim,
        )
        self.output_projection = nn.Linear(controller_dim, hidden_dim)

    def forward(
        self,
        style: Tensor,
        depth: Tensor,
        hidden_shape: torch.Size,
    ) -> Tensor:
        if style.ndim != 2 or depth.ndim != 2 or style.shape[0] != depth.shape[0]:
            raise ValueError("style and depth must have matching [batch, dim] shapes")
        condition = self.condition_projection(torch.cat((style, depth), dim=-1))
        residual = self.output_projection(F.gelu(condition))
        shape = (residual.shape[0],) + (1,) * (len(hidden_shape) - 2) + (
            residual.shape[-1],
        )
        return residual.view(shape).expand(*hidden_shape[:-1], residual.shape[-1])


class ConceptLearner(nn.Module):
    """Learn explicit genre-centered concept directions.

    ``group_matrix`` is an optional ``[num_concepts, num_groups]`` artist-to-
    genre membership matrix.  Rows may be multi-hot or soft; they are
    normalized internally.  When omitted, every other concept is a peer.

    The model is intentionally additive and condition-only.  There is no text
    router, hidden-state router, replacement predictor, or runtime concept
    availability state.
    """

    def __init__(
        self,
        num_concepts: int,
        hidden_dim: int,
        style_dim: int,
        num_blocks: int,
        *,
        concept_name: str = "artist",
        controller_dim: Optional[int] = None,
        depth_dim: int = 32,
        initial_block_scale: float = 0.0,
        max_block_scale: float = 1.0,
        cap_intervention_rms: Optional[bool] = None,
        normalize_intervention: Optional[bool] = None,
        intervention_norm_epsilon: float = 1e-12,
        intervention_blocks: Optional[Sequence[int]] = None,
        group_matrix: Optional[Sequence[Sequence[float]] | Tensor] = None,
    ) -> None:
        super().__init__()
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        if hidden_dim <= 0 or style_dim <= 0 or num_blocks <= 0 or depth_dim <= 0:
            raise ValueError(
                "hidden_dim, style_dim, num_blocks and depth_dim must be positive"
            )
        if max_block_scale <= 0:
            raise ValueError("max_block_scale must be positive")
        if intervention_norm_epsilon <= 0:
            raise ValueError("intervention_norm_epsilon must be positive")
        if abs(float(initial_block_scale)) >= float(max_block_scale):
            raise ValueError(
                "abs(initial_block_scale) must be smaller than max_block_scale"
            )
        if not concept_name.strip():
            raise ValueError("concept_name must not be empty")

        self.num_concepts = int(num_concepts)
        self.hidden_dim = int(hidden_dim)
        self.style_dim = int(style_dim)
        self.num_blocks = int(num_blocks)
        self.concept_name = concept_name
        self.max_block_scale = float(max_block_scale)
        if (
            cap_intervention_rms is not None
            and normalize_intervention is not None
            and bool(cap_intervention_rms) != bool(normalize_intervention)
        ):
            raise ValueError(
                "cap_intervention_rms and legacy normalize_intervention disagree"
            )
        if cap_intervention_rms is None:
            cap_intervention_rms = (
                True
                if normalize_intervention is None
                else bool(normalize_intervention)
            )
        self.cap_intervention_rms = bool(cap_intervention_rms)
        # Attribute retained for code/configs written before the no-amplification
        # cap replaced equal-RMS normalization.
        self.normalize_intervention = self.cap_intervention_rms
        self.intervention_norm_epsilon = float(intervention_norm_epsilon)

        controller_dim = int(controller_dim or hidden_dim)
        if controller_dim <= 0:
            raise ValueError("controller_dim must be positive")

        self.style_bank = ConceptStyleBank(num_concepts, style_dim)
        self.depth_embeddings = nn.Embedding(num_blocks, depth_dim)
        nn.init.normal_(self.depth_embeddings.weight, std=0.02)
        # Store an unconstrained raw value while exposing a bounded effective
        # scale in ``effective_block_scales``.  Multiplying ``atanh`` by the
        # bound makes the derivative at zero equal to one, so zero-init keeps
        # its useful gate gradient without permitting unbounded intervention.
        initial_ratio = float(initial_block_scale) / self.max_block_scale
        raw_initial_scale = self.max_block_scale * math.atanh(initial_ratio)
        self.block_scales = nn.Parameter(
            torch.full((num_blocks,), raw_initial_scale)
        )
        self.style_decoder = _ConditionOnlyStyleDecoder(
            hidden_dim,
            style_dim,
            depth_dim,
            controller_dim,
        )

        enabled = torch.zeros(num_blocks, dtype=torch.bool)
        if intervention_blocks is None:
            enabled.fill_(True)
        else:
            for block_index in intervention_blocks:
                self._validate_block_index(block_index)
                enabled[block_index] = True
        self.register_buffer("intervention_mask", enabled)
        self.register_buffer(
            "group_matrix",
            self._prepare_group_matrix(group_matrix),
        )

    def extra_repr(self) -> str:
        return (
            f"concept_name={self.concept_name!r}, "
            f"num_concepts={self.num_concepts}, num_blocks={self.num_blocks}, "
            f"max_block_scale={self.max_block_scale}, "
            f"cap_intervention_rms={self.cap_intervention_rms}"
        )

    @property
    def effective_block_scales(self) -> Tensor:
        """Return the bounded per-block intervention gates.

        ``block_scales`` remains the checkpoint parameter name for backward
        compatibility, but it is an unconstrained raw value.  Callers and
        monitoring code should use this property for the scale that is
        actually applied to hidden states.
        """

        bound = self.block_scales.new_tensor(self.max_block_scale)
        return bound * torch.tanh(self.block_scales / bound)

    def _validate_block_index(self, block_index: int) -> None:
        if not isinstance(block_index, int):
            raise TypeError("block_index must be a Python int")
        if not 0 <= block_index < self.num_blocks:
            raise IndexError(
                f"block_index must be in [0, {self.num_blocks}), got {block_index}"
            )

    def _prepare_group_matrix(
        self,
        value: Optional[Sequence[Sequence[float]] | Tensor],
    ) -> Tensor:
        if value is None:
            return torch.empty(self.num_concepts, 0)
        matrix = torch.as_tensor(value, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != self.num_concepts:
            raise ValueError(
                "group_matrix must have shape "
                f"[{self.num_concepts}, num_groups], got {tuple(matrix.shape)}"
            )
        if matrix.shape[1] == 0 or (matrix < 0).any():
            raise ValueError("group_matrix must contain non-negative group weights")
        totals = matrix.sum(dim=-1, keepdim=True)
        return torch.where(totals > 0, matrix / totals.clamp_min(1e-12), matrix)

    @staticmethod
    def _normalise(weights: Tensor) -> Tensor:
        totals = weights.sum(dim=-1, keepdim=True)
        return torch.where(
            totals > 0,
            weights / totals.clamp_min(torch.finfo(weights.dtype).tiny),
            torch.zeros_like(weights),
        )

    def _ids_to_weights(self, concept_ids: ConceptIds) -> Tensor:
        device = self.style_bank.embeddings.weight.device
        ids = torch.as_tensor(concept_ids, device=device)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        if ids.ndim != 1:
            raise ValueError("concept_ids must have shape [batch]")
        ids = ids.long()
        if ids.numel() and ((ids < 0).any() or (ids >= self.num_concepts).any()):
            raise IndexError(f"concept ids must be in [0, {self.num_concepts})")
        return F.one_hot(ids, num_classes=self.num_concepts).to(
            dtype=self.style_bank.embeddings.weight.dtype
        )

    def _peer_weights(self, weights: Tensor) -> Tensor:
        """Return genre-matched peers while excluding selected concepts."""

        selected = weights.gt(0)
        if self.group_matrix.shape[1] > 0:
            groups = self.group_matrix.to(weights)
            profile = self._normalise(weights @ groups)
            affinity = profile @ groups.transpose(0, 1)
        else:
            affinity = torch.ones_like(weights)
        affinity = affinity.masked_fill(selected, 0.0)
        peers = self._normalise(affinity)

        missing = peers.sum(dim=-1, keepdim=True).eq(0) & weights.sum(
            dim=-1, keepdim=True
        ).gt(0)
        if missing.any():
            fallback = self._normalise((~selected).to(weights))
            peers = torch.where(missing, fallback, peers)
        return peers

    def prepare_condition(
        self,
        *,
        concept_ids: Optional[ConceptIds] = None,
        concept_weights: Optional[Tensor] = None,
        direction: Union[float, Tensor] = 1.0,
        batch_size: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> ConceptCondition:
        """Prepare an explicit positive, negative, or null condition."""

        supplied = int(concept_ids is not None) + int(concept_weights is not None)
        if supplied > 1:
            raise ValueError("supply concept_ids or concept_weights, not both")

        target_device = torch.device(device) if device is not None else None
        if concept_ids is not None:
            weights = self._ids_to_weights(concept_ids)
        elif concept_weights is not None:
            weights = torch.as_tensor(
                concept_weights,
                device=self.style_bank.embeddings.weight.device,
                dtype=self.style_bank.embeddings.weight.dtype,
            )
            if weights.ndim != 2 or weights.shape[-1] != self.num_concepts:
                raise ValueError(
                    f"concept_weights must have shape [batch, {self.num_concepts}]"
                )
            if (weights < 0).any():
                raise ValueError("concept_weights must be non-negative")
            weights = self._normalise(weights)
        else:
            if batch_size is None or batch_size <= 0:
                raise ValueError("null conditions require a positive batch_size")
            weights = self.style_bank.embeddings.weight.new_zeros(
                int(batch_size),
                self.num_concepts,
            )

        if target_device is not None:
            weights = weights.to(target_device)
        direction_tensor = torch.as_tensor(
            direction,
            device=weights.device,
            dtype=weights.dtype,
        )
        if direction_tensor.ndim == 0:
            direction_tensor = direction_tensor.expand(weights.shape[0])
        if direction_tensor.shape != (weights.shape[0],):
            raise ValueError("direction must be scalar or have shape [batch]")
        active = weights.sum(dim=-1).gt(0).to(weights)
        return ConceptCondition(
            weights=weights,
            peer_weights=self._peer_weights(weights),
            direction=direction_tensor,
            active=active,
        )

    def positive_condition(self, concept_ids: ConceptIds) -> ConceptCondition:
        return self.prepare_condition(concept_ids=concept_ids, direction=1.0)

    def suppression_condition(self, concept_ids: ConceptIds) -> ConceptCondition:
        return self.prepare_condition(concept_ids=concept_ids, direction=-1.0)

    def null_condition(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device | str] = None,
    ) -> ConceptCondition:
        return self.prepare_condition(batch_size=batch_size, direction=0.0, device=device)

    @staticmethod
    def _expand_batch_rows(
        value: Tensor,
        target_batch: int,
        repeat_interleave: int,
    ) -> tuple[Tensor, int, int]:
        """Align sample rows with MusicGen codebooks and optional CFG copies."""

        if value.shape[0] == target_batch:
            return value, 1, 1
        base_batch = value.shape[0]
        if target_batch % base_batch:
            raise ValueError(
                "condition batch must match or evenly divide hidden-state batch"
            )
        effective_interleave = 1
        expanded = value
        if (
            repeat_interleave > 1
            and base_batch * repeat_interleave <= target_batch
            and target_batch % (base_batch * repeat_interleave) == 0
        ):
            expanded = value.repeat_interleave(repeat_interleave, dim=0)
            effective_interleave = repeat_interleave
        copies = target_batch // expanded.shape[0]
        if copies > 1:
            expanded = expanded.repeat((copies,) + (1,) * (value.ndim - 1))
        return expanded, effective_interleave, copies

    def _batch_scale(
        self,
        scale: Union[float, Tensor],
        hidden_state: Tensor,
        repeat_interleave: int,
    ) -> Tensor:
        value = torch.as_tensor(
            scale,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        if value.ndim == 0:
            return value
        if value.ndim != 1:
            raise ValueError("intervention_scale must be scalar or [batch]")
        value, _, _ = self._expand_batch_rows(
            value,
            hidden_state.shape[0],
            repeat_interleave,
        )
        return value.view((hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1))

    def _cap_to_hidden_rms(
        self,
        residual: Tensor,
        hidden_state: Tensor,
    ) -> Tensor:
        """Clip each token's residual RMS without ever increasing it.

        Clipping over the hidden dimension keeps teacher-forced sequence
        forwards and single-token cached generation consistent. The incoming
        hidden magnitude is detached: it is a safety reference, not a path
        through which a trainable generator can game the cap. In contrast to
        equal-RMS normalization, this operation leaves small residuals exactly
        as they are.
        """

        if not self.cap_intervention_rms:
            return residual
        epsilon = self.intervention_norm_epsilon
        residual_float = residual.float()
        residual_rms = (
            residual_float.square()
            .mean(dim=-1, keepdim=True)
            .clamp_min(epsilon)
            .sqrt()
        )
        hidden_rms = (
            hidden_state.detach()
            .float()
            .square()
            .mean(dim=-1, keepdim=True)
            .clamp_min(epsilon)
            .sqrt()
        )
        shrink = (hidden_rms / residual_rms).clamp(max=1.0)
        capped_residual = residual_float * shrink
        return capped_residual.to(dtype=residual.dtype)

    def forward(
        self,
        hidden_state: Tensor,
        block_index: int,
        condition: ConceptCondition,
        *,
        intervention_scale: Union[float, Tensor] = 1.0,
        batch_repeat_interleave: int = 1,
        cfg_conditional_only: bool = False,
        return_details: bool = False,
    ) -> Tensor | Dict[str, Tensor]:
        self._validate_block_index(block_index)
        if hidden_state.ndim < 2 or hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden_state must end in hidden_dim={self.hidden_dim}, "
                f"got {tuple(hidden_state.shape)}"
            )
        if not bool(self.intervention_mask[block_index]):
            zero = torch.zeros_like(hidden_state)
            details = {
                "hidden_state": hidden_state,
                "concept_residual": zero,
                "capped_residual": zero,
                "target_residual": zero,
                "peer_residual": zero,
                "effective_scale": hidden_state.new_zeros(()),
                "intervention_delta": zero,
            }
            return details if return_details else hidden_state

        weights, effective_interleave, copies = self._expand_batch_rows(
            condition.weights,
            hidden_state.shape[0],
            batch_repeat_interleave,
        )
        peer_weights, _, _ = self._expand_batch_rows(
            condition.peer_weights,
            hidden_state.shape[0],
            effective_interleave,
        )
        direction, _, _ = self._expand_batch_rows(
            condition.direction,
            hidden_state.shape[0],
            effective_interleave,
        )
        active, _, _ = self._expand_batch_rows(
            condition.active,
            hidden_state.shape[0],
            effective_interleave,
        )
        weights = weights.to(hidden_state)
        peer_weights = peer_weights.to(hidden_state)
        direction = direction.to(hidden_state)
        active = active.to(hidden_state)

        depth = self.depth_embeddings.weight[block_index].to(hidden_state)
        depth = depth.unsqueeze(0).expand(hidden_state.shape[0], -1)
        target_style = self.style_bank(weights).to(hidden_state)
        peer_style = self.style_bank(peer_weights).to(hidden_state)
        target_residual = self.style_decoder(
            target_style,
            depth,
            hidden_state.shape,
        )
        peer_residual = self.style_decoder(
            peer_style,
            depth,
            hidden_state.shape,
        )
        concept_residual = target_residual - peer_residual
        row_shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        signed_residual = (
            active.view(row_shape)
            * direction.view(row_shape)
            * concept_residual
        )
        if cfg_conditional_only and copies >= 2:
            cfg_gate = hidden_state.new_ones(hidden_state.shape[0])
            cfg_gate[hidden_state.shape[0] // 2 :] = 0
            signed_residual = signed_residual * cfg_gate.view(row_shape)

        capped_residual = self._cap_to_hidden_rms(
            signed_residual,
            hidden_state,
        )

        layer_scale = self.effective_block_scales[block_index].to(hidden_state)
        external_scale = self._batch_scale(
            intervention_scale,
            hidden_state,
            effective_interleave,
        )
        effective_scale = layer_scale * external_scale
        intervention_delta = effective_scale * capped_residual
        changed = hidden_state + intervention_delta
        if not return_details:
            return changed
        return {
            "hidden_state": changed,
            "concept_residual": concept_residual,
            "capped_residual": capped_residual,
            "target_residual": target_residual,
            "peer_residual": peer_residual,
            "effective_scale": effective_scale,
            "intervention_delta": intervention_delta,
        }

    def regularization_losses(self) -> Dict[str, Tensor]:
        embeddings = self.style_bank.deterministic_embeddings
        identity = torch.eye(
            self.num_concepts,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        peer_weights = self._peer_weights(identity)
        centered = embeddings - peer_weights @ embeddings
        return {
            "style_bank_l2": embeddings.square().mean(),
            "concept_center": centered.mean(dim=0).square().mean(),
            "concept_spread_monitor": centered.square().mean(),
        }

    def compute_losses(
        self,
        *,
        style_bank_l2_weight: float = 0.0,
        concept_center_weight: float = 0.0,
    ) -> Dict[str, Tensor]:
        regularizers = self.regularization_losses()
        total = (
            regularizers["style_bank_l2"] * float(style_bank_l2_weight)
            + regularizers["concept_center"] * float(concept_center_weight)
        )
        return {**regularizers, "loss": total}


__all__ = ["ConceptCondition", "ConceptLearner", "ConceptStyleBank"]
