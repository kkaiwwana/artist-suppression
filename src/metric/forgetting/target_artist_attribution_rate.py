"""Target Artist Attribution Rate (AAR) metric."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

from ._utils import call_classifier, expand_artist_ids


class TargetArtistAttributionRate(Metric):
    """Fraction of generations classified as the target artist.

    For unlearning evaluation this is usually computed on generations produced
    with the target artist disabled, so lower is better. The classifier is an
    external, already-trained artist attribution model.

    Args:
        classifier: External classifier returning ``[batch, num_artists]`` logits.
            If ``None``, the input passed to ``update`` is treated as logits.
        target_artist_id: Target artist id, or one id per sample.
        sample_rate: Optional audio sample rate passed to classifiers that accept it.
        logits_key: Mapping key used when classifier returns a dict-like output.
        classifier_kwargs: Extra keyword arguments forwarded to the classifier.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        classifier: Any | None,
        target_artist_id: int | Tensor,
        *,
        sample_rate: int | None = None,
        logits_key: str = "logits",
        classifier_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.classifier = classifier
        self.target_artist_id = target_artist_id
        self.sample_rate = sample_rate
        self.logits_key = logits_key
        self.classifier_kwargs = dict(classifier_kwargs or {})

        self.add_state("target_count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        audio_or_logits: Tensor,
        target_artist_id: int | Tensor | None = None,
        *,
        sample_rate: int | None = None,
    ) -> None:
        logits = call_classifier(
            self.classifier,
            audio_or_logits,
            sample_rate=sample_rate if sample_rate is not None else self.sample_rate,
            logits_key=self.logits_key,
            forward_kwargs=self.classifier_kwargs,
        )
        preds = logits.argmax(dim=-1)
        target_ids = expand_artist_ids(
            self.target_artist_id if target_artist_id is None else target_artist_id,
            batch_size=preds.shape[0],
            device=preds.device,
        )
        self.target_count += (preds == target_ids).float().sum()
        self.total += torch.tensor(float(preds.numel()), device=self.total.device)

    def compute(self) -> Tensor:
        return self.target_count / self.total.clamp_min(1.0)
