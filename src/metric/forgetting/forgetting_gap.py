"""Forgetting Gap metric."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

from ._utils import call_classifier, expand_artist_ids


class ForgettingGap(Metric):
    """Mean drop of target-artist probability after applying forgetting.

    The metric compares a reference/control generation against a forgotten
    generation for the same prompt:

    ``gap = mean(p_classifier(target | reference) - p_classifier(target | forgotten))``

    Positive values mean the target artist became less attributable after the
    forgetting intervention.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        classifier: Any | None,
        target_artist_id: int | Tensor,
        *,
        sample_rate: int | None = None,
        from_logits: bool = True,
        logits_key: str = "logits",
        classifier_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.classifier = classifier
        self.target_artist_id = target_artist_id
        self.sample_rate = sample_rate
        self.from_logits = from_logits
        self.logits_key = logits_key
        self.classifier_kwargs = dict(classifier_kwargs or {})

        self.add_state("gap_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _target_probability(
        self,
        audio_or_scores: Tensor,
        target_artist_id: int | Tensor,
        *,
        sample_rate: int | None,
    ) -> Tensor:
        scores = call_classifier(
            self.classifier,
            audio_or_scores,
            sample_rate=sample_rate,
            logits_key=self.logits_key,
            forward_kwargs=self.classifier_kwargs,
        )
        probs = scores.softmax(dim=-1) if self.from_logits else scores
        target_ids = expand_artist_ids(
            target_artist_id,
            batch_size=probs.shape[0],
            device=probs.device,
        )
        return probs.gather(1, target_ids[:, None]).squeeze(1)

    def update(
        self,
        reference_audio_or_scores: Tensor,
        forgotten_audio_or_scores: Tensor,
        target_artist_id: int | Tensor | None = None,
        *,
        sample_rate: int | None = None,
    ) -> None:
        target_ids = self.target_artist_id if target_artist_id is None else target_artist_id
        sr = sample_rate if sample_rate is not None else self.sample_rate
        reference_prob = self._target_probability(
            reference_audio_or_scores, target_ids, sample_rate=sr
        )
        forgotten_prob = self._target_probability(
            forgotten_audio_or_scores, target_ids, sample_rate=sr
        )
        if reference_prob.shape != forgotten_prob.shape:
            raise ValueError(
                "Reference and forgotten batches must have the same batch size; "
                f"got {tuple(reference_prob.shape)} and {tuple(forgotten_prob.shape)}."
            )

        self.gap_sum += (reference_prob - forgotten_prob).sum()
        self.total += torch.tensor(float(reference_prob.numel()), device=self.total.device)

    def compute(self) -> Tensor:
        return self.gap_sum / self.total.clamp_min(1.0)
