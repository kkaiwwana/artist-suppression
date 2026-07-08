"""Tempo error metric."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

from ._audio import estimate_tempo_bpm, to_mono_batch


class TempoError(Metric):
    """Mean absolute tempo error in BPM.

    The reference can be either another waveform or a tensor of reference BPM
    values. If ``reference_is_bpm`` is ``None``, a 0-D/1-D tensor whose length
    equals the generated batch size is treated as BPM labels.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        *,
        sample_rate: int,
        frame_length: int = 2048,
        hop_length: int = 512,
        bpm_min: float = 30.0,
        bpm_max: float = 240.0,
        reference_is_bpm: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.reference_is_bpm = reference_is_bpm

        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _estimate(self, audio: Tensor, sample_rate: int) -> Tensor:
        return estimate_tempo_bpm(
            audio,
            sample_rate=sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            bpm_min=self.bpm_min,
            bpm_max=self.bpm_max,
        )

    def _reference_to_bpm(self, reference: Tensor, batch_size: int, sample_rate: int) -> Tensor:
        reference_tensor = torch.as_tensor(reference)
        is_bpm = self.reference_is_bpm
        if is_bpm is None:
            is_bpm = reference_tensor.ndim == 0 or (
                reference_tensor.ndim == 1 and reference_tensor.numel() == batch_size
            )

        if is_bpm:
            bpm = reference_tensor.to(dtype=torch.float32).flatten()
            if bpm.numel() == 1:
                bpm = bpm.repeat(batch_size)
            if bpm.numel() != batch_size:
                raise ValueError(
                    f"Expected {batch_size} reference BPM values, got {bpm.numel()}."
                )
            return bpm
        return self._estimate(reference_tensor, sample_rate)

    def update(
        self,
        generated_audio: Tensor,
        reference_audio_or_bpm: Tensor,
        *,
        sample_rate: int | None = None,
    ) -> None:
        sr = sample_rate if sample_rate is not None else self.sample_rate
        generated_bpm = self._estimate(generated_audio, sr)
        reference_bpm = self._reference_to_bpm(
            reference_audio_or_bpm, batch_size=generated_bpm.numel(), sample_rate=sr
        ).to(generated_bpm.device)

        if reference_bpm.numel() == 1 and generated_bpm.numel() > 1:
            reference_bpm = reference_bpm.repeat(generated_bpm.numel())
        if reference_bpm.numel() != generated_bpm.numel():
            # If the reference is a singleton audio waveform, broadcast it.
            reference_waveform_batch = to_mono_batch(reference_audio_or_bpm).shape[0]
            if reference_waveform_batch == 1:
                reference_bpm = reference_bpm.repeat(generated_bpm.numel())
            else:
                raise ValueError(
                    f"Expected {generated_bpm.numel()} reference tempos, got {reference_bpm.numel()}."
                )

        errors = (generated_bpm - reference_bpm).abs()
        self.error_sum += errors.sum()
        self.total += torch.tensor(float(errors.numel()), device=self.total.device)

    def compute(self) -> Tensor:
        return self.error_sum / self.total.clamp_min(1.0)
