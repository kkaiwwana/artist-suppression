"""Audio helpers shared by quality metrics.

The helpers intentionally rely only on PyTorch so that the metrics are usable in
the current project environment. For paper-grade evaluation, CLAP/FAD should
still be paired with the same pretrained encoders used by the baseline papers.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)


def to_mono_batch(audio: Tensor | Any) -> Tensor:
    """Convert audio to ``[batch, time]`` float tensor.

    Accepted shapes:
        - ``[time]`` for a single mono waveform
        - ``[batch, time]`` for batched mono waveforms
        - ``[batch, channels, time]`` for batched multi-channel waveforms
    """

    if not isinstance(audio, Tensor):
        audio = torch.as_tensor(audio)
    audio = audio.float()

    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.ndim == 2:
        return audio
    if audio.ndim == 3:
        return audio.mean(dim=1)
    raise ValueError(
        "Expected audio with shape [time], [batch, time], or [batch, channels, time], "
        f"got {tuple(audio.shape)}."
    )


def match_batch(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcast singleton batches, otherwise require equal batch sizes."""

    if first.shape[0] == second.shape[0]:
        return first, second
    if first.shape[0] == 1:
        return first.expand(second.shape[0], *first.shape[1:]), second
    if second.shape[0] == 1:
        return first, second.expand(first.shape[0], *second.shape[1:])
    raise ValueError(
        f"Batch sizes must match or be singleton, got {first.shape[0]} and {second.shape[0]}."
    )


def compute_chroma(
    audio: Tensor | Any,
    *,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 512,
    fmin: float = 32.70319566257483,
    eps: float = 1e-8,
) -> Tensor:
    """Compute a simple STFT chromagram with pitch classes ordered C..B.

    Returns:
        Tensor with shape ``[batch, 12, frames]``.
    """

    waveform = to_mono_batch(audio)
    if waveform.shape[-1] < n_fft:
        waveform = F.pad(waveform, (0, n_fft - waveform.shape[-1]))

    window = torch.hann_window(
        n_fft, device=waveform.device, dtype=waveform.dtype
    )
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )
    magnitude = spectrum.abs()
    batch, freq_bins, frames = magnitude.shape

    frequencies = torch.linspace(
        0,
        sample_rate / 2,
        freq_bins,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    valid = frequencies >= max(float(fmin), eps)
    if not bool(valid.any()):
        return torch.zeros(batch, 12, frames, device=waveform.device, dtype=waveform.dtype)

    valid_frequencies = frequencies[valid]
    midi = 69.0 + 12.0 * torch.log2(valid_frequencies / 440.0)
    pitch_classes = torch.round(midi).long().remainder(12)

    valid_magnitude = magnitude[:, valid, :]
    chroma = torch.zeros(batch, 12, frames, device=waveform.device, dtype=waveform.dtype)
    chroma.scatter_add_(
        dim=1,
        index=pitch_classes.view(1, -1, 1).expand(batch, -1, frames),
        src=valid_magnitude,
    )
    chroma = chroma / chroma.norm(dim=1, keepdim=True).clamp_min(eps)
    return chroma


def estimate_tempo_bpm(
    audio: Tensor | Any,
    *,
    sample_rate: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    bpm_min: float = 30.0,
    bpm_max: float = 240.0,
    eps: float = 1e-8,
) -> Tensor:
    """Estimate tempo using a lightweight onset-envelope autocorrelation.

    This is intentionally dependency-light. It is suitable for sanity checks and
    relative comparisons; production MIR evaluation can replace it with a
    stronger beat tracker while keeping the metric API unchanged.
    """

    waveform = to_mono_batch(audio)
    if waveform.shape[-1] < frame_length:
        waveform = F.pad(waveform, (0, frame_length - waveform.shape[-1]))

    frames = waveform.unfold(dimension=-1, size=frame_length, step=hop_length)
    if frames.shape[1] < 3:
        return torch.zeros(waveform.shape[0], device=waveform.device, dtype=waveform.dtype)

    energy = frames.pow(2).mean(dim=-1).sqrt()
    onset = torch.relu(energy[:, 1:] - energy[:, :-1])
    if onset.shape[1] < 3:
        return torch.zeros(waveform.shape[0], device=waveform.device, dtype=waveform.dtype)

    onset = onset - onset.mean(dim=1, keepdim=True)
    onset = onset / onset.std(dim=1, keepdim=True).clamp_min(eps)

    frame_rate = sample_rate / hop_length
    lag_min = max(1, int(math.floor(frame_rate * 60.0 / bpm_max)))
    lag_max = min(onset.shape[1] - 1, int(math.ceil(frame_rate * 60.0 / bpm_min)))
    if lag_max < lag_min:
        return torch.zeros(waveform.shape[0], device=waveform.device, dtype=waveform.dtype)

    lags = torch.arange(lag_min, lag_max + 1, device=waveform.device)
    scores = []
    for lag in lags.tolist():
        scores.append((onset[:, :-lag] * onset[:, lag:]).mean(dim=1))
    score_matrix = torch.stack(scores, dim=1)
    best_lag = lags[score_matrix.argmax(dim=1)].to(dtype=waveform.dtype)
    return 60.0 * frame_rate / best_lag


def _normalized_template(values: list[float], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    template = torch.tensor(values, device=device, dtype=dtype)
    template = template - template.mean()
    return template / template.norm().clamp_min(1e-8)


def key_template_matrix(*, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return 24 Krumhansl-Schmuckler key templates: 12 major then 12 minor."""

    major = _normalized_template(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        device=device,
        dtype=dtype,
    )
    minor = _normalized_template(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        device=device,
        dtype=dtype,
    )
    templates = [torch.roll(major, shifts=root, dims=0) for root in range(12)]
    templates.extend(torch.roll(minor, shifts=root, dims=0) for root in range(12))
    return torch.stack(templates, dim=0)


def chord_template_matrix(*, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return 24 simple triad templates: 12 major then 12 minor."""

    templates = []
    for mode in ("major", "minor"):
        for root in range(12):
            template = torch.zeros(12, device=device, dtype=dtype)
            intervals = (0, 4, 7) if mode == "major" else (0, 3, 7)
            for interval in intervals:
                template[(root + interval) % 12] = 1.0
            template = template - template.mean()
            template = template / template.norm().clamp_min(1e-8)
            templates.append(template)
    return torch.stack(templates, dim=0)


def estimate_key_ids(chroma: Tensor) -> Tensor:
    """Estimate one key id per sample from chroma.

    Id convention: ``0..11`` = major keys C..B, ``12..23`` = minor keys C..B.
    """

    pooled = chroma.mean(dim=-1)
    pooled = pooled - pooled.mean(dim=1, keepdim=True)
    pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-8)
    templates = key_template_matrix(device=chroma.device, dtype=chroma.dtype)
    return (pooled @ templates.T).argmax(dim=1)


def estimate_chord_ids(chroma: Tensor) -> Tensor:
    """Estimate one major/minor triad id per chroma frame."""

    frame_chroma = chroma.transpose(1, 2)
    frame_chroma = frame_chroma - frame_chroma.mean(dim=-1, keepdim=True)
    frame_chroma = frame_chroma / frame_chroma.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    templates = chord_template_matrix(device=chroma.device, dtype=chroma.dtype)
    return (frame_chroma @ templates.T).argmax(dim=-1)


def extract_tensor_output(output: Any, preferred_key: str = "embeddings") -> Tensor:
    """Extract tensor outputs from model/encoder return values."""

    if isinstance(output, Tensor):
        return output
    if isinstance(output, dict):
        for key in (preferred_key, "embedding", "embeddings", "features", "last_hidden_state"):
            if key in output:
                value = output[key]
                return value if isinstance(value, Tensor) else torch.as_tensor(value)
    if isinstance(output, (tuple, list)) and output:
        value = output[0]
        return value if isinstance(value, Tensor) else torch.as_tensor(value)
    return torch.as_tensor(output)
