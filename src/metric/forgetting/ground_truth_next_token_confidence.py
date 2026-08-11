"""Ground-Truth Next-Token Confidence (GT-NTC) metric."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


def _valid_label_mask(
    labels: Tensor,
    *,
    pad_token_id: int | None,
    ignore_index: int,
    time_mask: Tensor | None,
) -> Tensor:
    if labels.ndim != 3:
        raise ValueError(
            "labels must have shape [batch, codebook, time]; "
            f"got {tuple(labels.shape)}."
        )
    if labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise TypeError(f"labels must contain integer token ids; got {labels.dtype}.")
    valid = labels.ne(ignore_index)
    if pad_token_id is not None:
        valid &= labels.ne(pad_token_id)
    if time_mask is not None:
        if not isinstance(time_mask, Tensor):
            raise TypeError("time_mask must be a torch.Tensor when provided.")
        expected_shape = (labels.shape[0], labels.shape[2])
        if tuple(time_mask.shape) != expected_shape:
            raise ValueError(
                f"time_mask must have shape [batch, time] {expected_shape}; "
                f"got {tuple(time_mask.shape)}."
            )
        valid &= time_mask.to(device=labels.device, dtype=torch.bool)[:, None, :]
    return valid


def _ground_truth_token_probabilities(
    logits: Tensor,
    labels: Tensor,
    *,
    pad_token_id: int | None,
    ignore_index: int,
    time_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Return aligned token probabilities and their validity mask."""
    if logits.ndim != 4:
        raise ValueError(
            "logits must have shape [batch, codebook, time, vocabulary]; "
            f"got {tuple(logits.shape)}."
        )
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "logits and labels must align on [batch, codebook, time]; "
            f"got {tuple(logits.shape)} and {tuple(labels.shape)}."
        )
    if logits.shape[-1] == 0:
        raise ValueError("logits vocabulary dimension must be non-empty.")
    valid = _valid_label_mask(
        labels,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        time_mask=time_mask,
    )

    # Replacing ignored ids before gather avoids indexing with values such as
    # -100. Using logsumexp computes the selected softmax probabilities without
    # materializing a second tensor of full-vocabulary probabilities.
    safe_labels = labels.masked_fill(~valid, 0).long()
    # The metric often runs under mixed precision. Compute the probability in
    # float32 so genuinely small confidences (for example exp(-20)) do not
    # collapse to exactly zero in fp16.
    float_logits = logits.float()
    selected_logits = float_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    probabilities = (selected_logits - torch.logsumexp(float_logits, dim=-1)).exp()
    return probabilities, valid


@torch.no_grad()
def ground_truth_next_token_confidence_per_sample(
    logits: Tensor,
    labels: Tensor,
    pad_token_id: int | None = None,
    ignore_index: int = -100,
    time_mask: Tensor | None = None,
) -> Tensor:
    """Compute GT-NTC independently for every clip in a batch.

    Args:
        logits: Aligned next-token logits with shape ``[B, Q, T, V]``.
        labels: Ground-truth token ids with shape ``[B, Q, T]``.
        pad_token_id: Optional padding id to ignore.
        ignore_index: Ignored label value. Defaults to ``-100``.
        time_mask: Optional mask with shape ``[B, T]``. A true/non-zero entry
            includes that time position for every codebook, allowing evaluation
            to be restricted to the ground-truth continuation tail.

    Returns:
        A float64 tensor of shape ``[B]``. Samples with no valid tokens return
        zero.
    """
    probabilities, valid = _ground_truth_token_probabilities(
        logits,
        labels,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        time_mask=time_mask,
    )
    valid_float = valid.to(dtype=probabilities.dtype)
    confidence_sum = (probabilities * valid_float).double().sum(dim=(1, 2))
    token_count = valid.sum(dim=(1, 2))
    scores = confidence_sum / token_count.clamp_min(1)
    return scores.masked_fill(token_count.eq(0), float("nan"))


@torch.no_grad()
def ground_truth_next_token_confidence_from_token_losses_per_sample(
    token_losses: Tensor,
    labels: Tensor,
    pad_token_id: int | None = None,
    ignore_index: int = -100,
    time_mask: Tensor | None = None,
) -> Tensor:
    """Compute per-clip GT-NTC from existing unreduced cross entropy.

    For a ground-truth class, ``exp(-cross_entropy)`` is exactly its softmax
    probability. MusicGen already returns these float32 losses, so this path
    avoids another full-vocabulary reduction during training and evaluation.
    """

    if token_losses.ndim != 3 or token_losses.shape != labels.shape:
        raise ValueError(
            "token_losses and labels must have matching [batch, codebook, time] "
            f"shapes; got {tuple(token_losses.shape)} and {tuple(labels.shape)}."
        )
    valid = _valid_label_mask(
        labels,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        time_mask=time_mask,
    )
    probabilities = (-token_losses.float()).exp()
    confidence_sum = (probabilities * valid.to(probabilities)).sum(dim=(1, 2))
    token_count = valid.sum(dim=(1, 2))
    scores = confidence_sum.double() / token_count.clamp_min(1)
    return scores.masked_fill(token_count.eq(0), float("nan"))


@torch.no_grad()
def ground_truth_next_token_confidence_from_rollout_scores(
    generation_scores: Sequence[Tensor],
    ground_truth_labels: Tensor,
    *,
    prompt_frames: int,
    positions: Sequence[int] | Tensor,
    pad_token_id: int | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Score GT tokens under an autoregressive free-generation trajectory.

    ``generation_scores`` must be the per-step processed scores returned by
    MusicGen generation with ``output_scores=True``. At rollout step ``s``,
    each row is conditioned on the model tokens sampled at all earlier steps.
    For raw continuation frame ``r`` and codebook ``q``, MusicGen's delay
    pattern predicts the corresponding GT token at rollout step ``r + q``.

    The result has shape ``[B, P]`` and averages valid codebooks at each of the
    requested one-based continuation ``positions``. These are probabilities
    under the actual sampling distribution, after generation processors such
    as classifier-free guidance, temperature, and top-k filtering.
    """

    labels = torch.as_tensor(ground_truth_labels)
    if labels.ndim != 3:
        raise ValueError(
            "ground_truth_labels must have shape [batch, codebook, time]; "
            f"got {tuple(labels.shape)}"
        )
    if labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise TypeError("ground_truth_labels must contain integer token ids")
    if prompt_frames <= 0:
        raise ValueError("prompt_frames must be positive")
    requested = torch.as_tensor(positions, dtype=torch.long).flatten()
    if requested.numel() == 0 or bool((requested <= 0).any()):
        raise ValueError("positions must contain positive one-based indices")
    if int(prompt_frames + requested.max().item()) > labels.shape[-1]:
        raise ValueError("requested rollout position exceeds ground-truth labels")

    batch_size, num_codebooks, _ = labels.shape
    required_steps = int(requested.max().item()) + num_codebooks - 1
    if len(generation_scores) < required_steps:
        raise ValueError(
            f"rollout scores contain {len(generation_scores)} steps, but "
            f"{required_steps} are required"
        )

    result = torch.empty(
        batch_size,
        requested.numel(),
        device=labels.device,
        dtype=torch.float32,
    )
    batch_offsets = torch.arange(batch_size, device=labels.device) * num_codebooks
    for column, position in enumerate(requested.tolist()):
        raw_frame = prompt_frames + int(position) - 1
        probability_sum = result.new_zeros(batch_size)
        valid_count = torch.zeros(batch_size, device=labels.device, dtype=torch.long)
        for codebook in range(num_codebooks):
            step = int(position) - 1 + codebook
            step_scores = torch.as_tensor(generation_scores[step])
            if step_scores.ndim == 3:
                if step_scores.shape[:2] != (batch_size, num_codebooks):
                    raise ValueError(
                        "3D rollout scores must have shape [batch, codebook, vocab]"
                    )
                step_scores = step_scores.flatten(0, 1)
            if step_scores.ndim != 2 or step_scores.shape[0] != batch_size * num_codebooks:
                raise ValueError(
                    "rollout scores must have shape [batch * codebook, vocabulary]"
                )
            target = labels[:, codebook, raw_frame].to(step_scores.device).long()
            valid = target.ne(ignore_index)
            if pad_token_id is not None:
                valid &= target.ne(pad_token_id)
            if bool(valid.any()):
                valid_targets = target[valid]
                if bool(
                    ((valid_targets < 0) | (valid_targets >= step_scores.shape[-1])).any()
                ):
                    raise ValueError("ground-truth token is outside score vocabulary")
                rows = (batch_offsets + codebook).to(step_scores.device)[valid]
                selected_scores = step_scores.index_select(0, rows).float()
                selected_targets = valid_targets.unsqueeze(-1)
                probabilities = (
                    selected_scores.gather(1, selected_targets).squeeze(1)
                    - torch.logsumexp(selected_scores, dim=-1)
                ).exp()
                probability_sum[valid.to(probability_sum.device)] += probabilities.to(
                    probability_sum
                )
                valid_count[valid.to(valid_count.device)] += 1
        values = probability_sum / valid_count.clamp_min(1)
        result[:, column] = values.masked_fill(valid_count.eq(0), float("nan"))
    return result


class GroundTruthNextTokenConfidence(Metric):
    """Mean model probability assigned to the aligned ground-truth token.

    The metric is intended for teacher-forced evaluation of autoregressive music
    models. ``logits`` must have shape ``[batch, codebook, time, vocabulary]``
    and ``labels`` must contain the corresponding next-token ids with shape
    ``[batch, codebook, time]``. Padding and ignored positions do not contribute
    to either the confidence sum or token count.

    Args:
        pad_token_id: Optional padding token id to exclude from the metric.
        ignore_index: Label value used for ignored positions. Defaults to ``-100``.
        kwargs: Additional keyword arguments forwarded to :class:`torchmetrics.Metric`.
    """

    display_name = "GT-NTC"
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        *,
        pad_token_id: int | None = None,
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        self.add_state(
            "confidence_sum",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "token_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        time_mask: Tensor | None = None,
        from_token_losses: bool = False,
    ) -> None:
        """Accumulate ground-truth next-token probabilities."""
        if from_token_losses:
            if logits.ndim != 3 or logits.shape != labels.shape:
                raise ValueError(
                    "token_losses and labels must have matching "
                    "[batch, codebook, time] shapes"
                )
            valid = _valid_label_mask(
                labels,
                pad_token_id=self.pad_token_id,
                ignore_index=self.ignore_index,
                time_mask=time_mask,
            )
            probabilities = (-logits.float()).exp()
            self.confidence_sum += probabilities[valid].sum().double()
            self.token_count += valid.sum().to(dtype=torch.long)
            return
        probabilities, valid = _ground_truth_token_probabilities(
            logits,
            labels,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
            time_mask=time_mask,
        )
        self.confidence_sum += probabilities[valid].double().sum()
        self.token_count += valid.sum().to(dtype=torch.long)

    @torch.no_grad()
    def update_from_token_losses(
        self,
        token_losses: Tensor,
        labels: Tensor,
        *,
        time_mask: Tensor | None = None,
    ) -> None:
        """Accumulate from an already-computed unreduced cross entropy."""

        # Calling the wrapped public update keeps TorchMetrics' update_count
        # and Lightning epoch-reset bookkeeping correct.
        self.update(
            token_losses,
            labels,
            time_mask=time_mask,
            from_token_losses=True,
        )

    def compute(self) -> Tensor:
        """Return mean confidence, or NaN when no valid token was observed."""
        value = self.confidence_sum / self.token_count.clamp_min(1)
        return value.masked_fill(self.token_count.eq(0), float("nan"))


__all__ = [
    "GroundTruthNextTokenConfidence",
    "ground_truth_next_token_confidence_from_token_losses_per_sample",
    "ground_truth_next_token_confidence_from_rollout_scores",
    "ground_truth_next_token_confidence_per_sample",
]
