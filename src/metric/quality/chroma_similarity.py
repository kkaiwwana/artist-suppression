"""Chroma similarity metric."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

from ._audio import compute_chroma, match_batch


class ChromaSimilarity(Metric):
    """Mean cosine similarity between generated/reference chroma profiles."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int = 4096,
        hop_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.add_state("similarity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        generated_audio: Tensor,
        reference_audio: Tensor,
        *,
        sample_rate: int | None = None,
    ) -> None:
        sr = sample_rate if sample_rate is not None else self.sample_rate
        generated_chroma = compute_chroma(
            generated_audio, sample_rate=sr, n_fft=self.n_fft, hop_length=self.hop_length
        ).mean(dim=-1)
        reference_chroma = compute_chroma(
            reference_audio, sample_rate=sr, n_fft=self.n_fft, hop_length=self.hop_length
        ).mean(dim=-1)
        generated_chroma, reference_chroma = match_batch(generated_chroma, reference_chroma)
        similarities = F.cosine_similarity(generated_chroma, reference_chroma, dim=-1)
        self.similarity_sum += similarities.sum()
        self.total += torch.tensor(float(similarities.numel()), device=self.total.device)

    def compute(self) -> Tensor:
        return self.similarity_sum / self.total.clamp_min(1.0)
