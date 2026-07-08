"""Other Artist Accuracy retention metric."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

from ._utils import call_classifier, normalize_ignored_ids


class OtherArtistAccuracy(Metric):
    """Classifier accuracy on non-forgotten artists.

    This retention metric should be evaluated on prompts/audio belonging to
    artists outside the forget set. Samples whose labels are in
    ``target_artist_id`` or ``ignored_artist_ids`` are skipped.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        classifier: Any | None,
        *,
        target_artist_id: int | list[int] | tuple[int, ...] | Tensor | None = None,
        ignored_artist_ids: list[int] | tuple[int, ...] | Tensor | None = None,
        sample_rate: int | None = None,
        logits_key: str = "logits",
        classifier_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.classifier = classifier
        self.target_artist_id = target_artist_id
        self.ignored_artist_ids = ignored_artist_ids
        self.sample_rate = sample_rate
        self.logits_key = logits_key
        self.classifier_kwargs = dict(classifier_kwargs or {})

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        audio_or_logits: Tensor,
        artist_ids: Tensor,
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
        labels = torch.as_tensor(artist_ids, dtype=torch.long, device=logits.device).flatten()
        if labels.numel() != logits.shape[0]:
            raise ValueError(
                f"Expected {logits.shape[0]} labels, got shape {tuple(labels.shape)}."
            )

        valid = torch.ones_like(labels, dtype=torch.bool)
        ignored = normalize_ignored_ids(
            target_artist_id=self.target_artist_id,
            ignored_artist_ids=self.ignored_artist_ids,
            device=logits.device,
        )
        if ignored is not None:
            valid = ~torch.isin(labels, ignored)

        if not valid.any():
            return

        preds = logits.argmax(dim=-1)
        self.correct += (preds[valid] == labels[valid]).float().sum()
        self.total += torch.tensor(float(valid.sum().item()), device=self.total.device)

    def compute(self) -> Tensor:
        return self.correct / self.total.clamp_min(1.0)
