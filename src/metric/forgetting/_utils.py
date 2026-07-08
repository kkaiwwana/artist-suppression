"""Shared utilities for classifier-backed forgetting metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


def extract_tensor_output(output: Any, preferred_key: str = "logits") -> Tensor:
    """Extract a tensor from common model output formats.

    The external artist classifier can be a plain ``nn.Module`` returning logits,
    a HuggingFace-style mapping containing ``logits``, or a tuple/list whose
    first item is the score tensor.
    """

    if isinstance(output, Tensor):
        return output

    if isinstance(output, Mapping):
        candidate_keys = (preferred_key, "logits", "scores", "preds", "prediction")
        for key in candidate_keys:
            if key in output:
                value = output[key]
                if not isinstance(value, Tensor):
                    value = torch.as_tensor(value)
                return value
        raise KeyError(
            f"Could not find a tensor output in mapping. Tried keys: {candidate_keys}."
        )

    if isinstance(output, (tuple, list)) and output:
        value = output[0]
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value)
        return value

    raise TypeError(
        "Classifier output must be a Tensor, mapping with logits/scores, or tuple/list."
    )


def call_classifier(
    classifier: Any,
    audio_or_logits: Tensor,
    *,
    sample_rate: int | None = None,
    logits_key: str = "logits",
    forward_kwargs: dict[str, Any] | None = None,
) -> Tensor:
    """Run an external classifier and return a ``[batch, num_artists]`` tensor.

    If ``classifier`` is ``None``, ``audio_or_logits`` is treated as precomputed
    logits. This is useful for quick ablations and unit tests, while the normal
    project path is to pass the trained artist classifier at metric init time.
    """

    if classifier is None:
        logits = extract_tensor_output(audio_or_logits, preferred_key=logits_key)
    else:
        kwargs = dict(forward_kwargs or {})
        was_training = getattr(classifier, "training", None)
        if hasattr(classifier, "eval"):
            classifier.eval()

        with torch.no_grad():
            try:
                if sample_rate is None:
                    output = classifier(audio_or_logits, **kwargs)
                else:
                    output = classifier(audio_or_logits, sample_rate=sample_rate, **kwargs)
            except TypeError:
                # Some user classifiers expose forward(audio) and do not accept
                # sample_rate. Retrying keeps the metric easy to use with both.
                output = classifier(audio_or_logits, **kwargs)

        if was_training is not None and hasattr(classifier, "train"):
            classifier.train(was_training)
        logits = extract_tensor_output(output, preferred_key=logits_key)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(
            f"Expected classifier logits with shape [batch, num_artists], got {tuple(logits.shape)}."
        )
    return logits


def expand_artist_ids(
    artist_id: int | Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Return one artist id per batch item."""

    ids = torch.as_tensor(artist_id, dtype=torch.long, device=device)
    if ids.ndim == 0:
        ids = ids.repeat(batch_size)
    elif ids.numel() == 1:
        ids = ids.reshape(1).repeat(batch_size)
    elif ids.shape[0] != batch_size:
        raise ValueError(
            f"Expected {batch_size} artist ids, got shape {tuple(ids.shape)}."
        )
    else:
        ids = ids.reshape(batch_size)
    return ids


def normalize_ignored_ids(
    *,
    target_artist_id: int | list[int] | tuple[int, ...] | Tensor | None = None,
    ignored_artist_ids: list[int] | tuple[int, ...] | Tensor | None = None,
    device: torch.device,
) -> Tensor | None:
    """Merge target/ignored artist ids into a single tensor."""

    pieces: list[Tensor] = []
    if target_artist_id is not None:
        pieces.append(torch.as_tensor(target_artist_id, dtype=torch.long, device=device).flatten())
    if ignored_artist_ids is not None:
        pieces.append(torch.as_tensor(ignored_artist_ids, dtype=torch.long, device=device).flatten())
    if not pieces:
        return None
    return torch.unique(torch.cat(pieces))
