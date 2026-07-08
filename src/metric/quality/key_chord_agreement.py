"""Key and chord agreement metrics."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

from ._audio import compute_chroma, estimate_chord_ids, estimate_key_ids, match_batch


class KeyChordAgreement(Metric):
    """Agreement of estimated key and frame-level major/minor triads."""

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

        self.add_state("key_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("key_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("chord_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("chord_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

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
        )
        reference_chroma = compute_chroma(
            reference_audio, sample_rate=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        generated_chroma, reference_chroma = match_batch(generated_chroma, reference_chroma)

        generated_key = estimate_key_ids(generated_chroma)
        reference_key = estimate_key_ids(reference_chroma)
        self.key_correct += (generated_key == reference_key).float().sum()
        self.key_total += torch.tensor(float(generated_key.numel()), device=self.key_total.device)

        generated_chords = estimate_chord_ids(generated_chroma)
        reference_chords = estimate_chord_ids(reference_chroma)
        frame_count = min(generated_chords.shape[1], reference_chords.shape[1])
        if frame_count > 0:
            chord_matches = (
                generated_chords[:, :frame_count] == reference_chords[:, :frame_count]
            ).float()
            self.chord_correct += chord_matches.sum()
            self.chord_total += torch.tensor(float(chord_matches.numel()), device=self.chord_total.device)

    def compute(self) -> dict[str, Tensor]:
        return {
            "key_agreement": self.key_correct / self.key_total.clamp_min(1.0),
            "chord_agreement": self.chord_correct / self.chord_total.clamp_min(1.0),
        }


class KeyAgreement(KeyChordAgreement):
    """Scalar wrapper returning only key agreement."""

    def compute(self) -> Tensor:
        return self.key_correct / self.key_total.clamp_min(1.0)


class ChordAgreement(KeyChordAgreement):
    """Scalar wrapper returning only frame-level chord agreement."""

    def compute(self) -> Tensor:
        return self.chord_correct / self.chord_total.clamp_min(1.0)
