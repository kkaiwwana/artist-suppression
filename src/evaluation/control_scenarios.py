"""Deterministic cohorts and control plans for epoch-end evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


SCENARIO_NAMES = (
    "no_control",
    "enhance_target",
    "suppress_single_target",
    "suppress_multiple_target",
    "suppress_single_others",
    "suppress_multiple_others",
)

SCENARIO_LABELS = {
    "no_control": "No Control",
    "enhance_target": "Enhance Target",
    "suppress_single_target": "Suppress Single Target",
    "suppress_multiple_target": "Suppress Multiple Target",
    "suppress_single_others": "Suppress Single Others",
    "suppress_multiple_others": "Suppress Multiple Others",
}

EVALUATION_METRICS = (
    "gt_token_confidence",
    "target_attribution_rate",
    "target_artist_confidence",
    "target_artist_rank",
    "mert_similarity",
    "clap_similarity",
    "passt_kl",
    "fad",
)

METRIC_LABELS = {
    "gt_token_confidence": "GT Token Confidence",
    "target_attribution_rate": "Target Attribution Rate",
    "target_artist_confidence": "Target Artist Confidence",
    "target_artist_rank": "Target Artist Rank",
    "mert_similarity": "MERT Similarity",
    "clap_similarity": "CLAP Similarity",
    "passt_kl": "KL",
    "fad": "FAD",
}


@dataclass(frozen=True)
class ScenarioCondition:
    """One of the six matched control conditions for a batch."""

    name: str
    direction: float
    weights: Tensor | None
    artist_sets: tuple[tuple[int, ...], ...]

    @property
    def label(self) -> str:
        return SCENARIO_LABELS[self.name]


@dataclass(frozen=True)
class CohortItem:
    """A fixed dataset row and its target artist identity."""

    dataset_index: int
    artist_key: str
    artist_index: int


@dataclass(frozen=True)
class MeanStd:
    """Population summary over the evaluation cohort."""

    mean: float
    std: float
    count: int


def _one_hot_weights(ids: Tensor, num_concepts: int) -> Tensor:
    weights = torch.zeros(
        ids.shape[0],
        num_concepts,
        device=ids.device,
        dtype=torch.float32,
    )
    return weights.scatter_(1, ids.long().unsqueeze(1), 1.0)


def _sample_others(
    target: int,
    *,
    count: int,
    num_concepts: int,
    generator: torch.Generator,
) -> list[int]:
    candidates = torch.cat(
        (torch.arange(target), torch.arange(target + 1, num_concepts))
    )
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[:count]].tolist()


def _sets_to_weights(
    artist_sets: Sequence[Sequence[int]],
    *,
    num_concepts: int,
    device: torch.device,
) -> Tensor:
    weights = torch.zeros(
        len(artist_sets), num_concepts, device=device, dtype=torch.float32
    )
    for row, artist_ids in enumerate(artist_sets):
        if not artist_ids:
            raise ValueError("a controlled artist set cannot be empty")
        ids = torch.as_tensor(artist_ids, device=device, dtype=torch.long)
        weights[row, ids] = 1.0 / float(ids.numel())
    return weights


def build_control_scenarios(
    target_ids: Tensor | Sequence[int],
    *,
    num_concepts: int,
    seed: int,
    multi_min_artists: int = 2,
    multi_max_artists: int = 5,
) -> tuple[ScenarioCondition, ...]:
    """Build the six matched conditions with random, reproducible 2--5 sets.

    ``suppress_multiple_target`` always contains the row's target artist.
    ``suppress_multiple_others`` never contains it. All multi-hot rows sum to
    one so changing the number of artists does not silently scale the control.
    """

    ids = torch.as_tensor(target_ids, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("target_ids must be a non-empty vector")
    if num_concepts < 3:
        raise ValueError("six-scenario evaluation needs at least three artists")
    if bool(((ids < 0) | (ids >= num_concepts)).any()):
        raise ValueError("target_ids contain an out-of-range artist index")
    if multi_min_artists < 2 or multi_max_artists < multi_min_artists:
        raise ValueError("multi-artist bounds must satisfy 2 <= min <= max")

    # A target-inclusive set can contain at most all concepts, whereas an
    # others-only set can contain at most num_concepts - 1.
    max_target_total = min(int(multi_max_artists), num_concepts)
    max_other_total = min(int(multi_max_artists), num_concepts - 1)
    min_target_total = min(int(multi_min_artists), max_target_total)
    min_other_total = min(int(multi_min_artists), max_other_total)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    target_sets = [(int(value),) for value in ids.cpu().tolist()]
    single_other_sets: list[tuple[int, ...]] = []
    multi_target_sets: list[tuple[int, ...]] = []
    multi_other_sets: list[tuple[int, ...]] = []
    for target in ids.cpu().tolist():
        single_other = _sample_others(
            int(target), count=1, num_concepts=num_concepts, generator=generator
        )
        single_other_sets.append(tuple(single_other))

        target_total = int(
            torch.randint(
                min_target_total,
                max_target_total + 1,
                (1,),
                generator=generator,
            ).item()
        )
        target_others = _sample_others(
            int(target),
            count=target_total - 1,
            num_concepts=num_concepts,
            generator=generator,
        )
        multi_target_sets.append((int(target), *target_others))

        other_total = int(
            torch.randint(
                min_other_total,
                max_other_total + 1,
                (1,),
                generator=generator,
            ).item()
        )
        multi_other_sets.append(
            tuple(
                _sample_others(
                    int(target),
                    count=other_total,
                    num_concepts=num_concepts,
                    generator=generator,
                )
            )
        )

    device = ids.device
    target_weights = _one_hot_weights(ids.to(device), num_concepts)
    single_other_weights = _sets_to_weights(
        single_other_sets, num_concepts=num_concepts, device=device
    )
    multi_target_weights = _sets_to_weights(
        multi_target_sets, num_concepts=num_concepts, device=device
    )
    multi_other_weights = _sets_to_weights(
        multi_other_sets, num_concepts=num_concepts, device=device
    )
    empty_sets = tuple(() for _ in range(ids.numel()))
    return (
        ScenarioCondition("no_control", 0.0, None, empty_sets),
        ScenarioCondition("enhance_target", 1.0, target_weights, tuple(target_sets)),
        ScenarioCondition(
            "suppress_single_target", -1.0, target_weights, tuple(target_sets)
        ),
        ScenarioCondition(
            "suppress_multiple_target",
            -1.0,
            multi_target_weights,
            tuple(multi_target_sets),
        ),
        ScenarioCondition(
            "suppress_single_others",
            -1.0,
            single_other_weights,
            tuple(single_other_sets),
        ),
        ScenarioCondition(
            "suppress_multiple_others",
            -1.0,
            multi_other_weights,
            tuple(multi_other_sets),
        ),
    )


def _record_artist_key(record: Mapping[str, Any]) -> str:
    for key in ("artist_key", "artist_id", "artist_name"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    raise ValueError("dataset record has no artist identity")


def select_balanced_cohort(
    records: Sequence[Mapping[str, Any]],
    artist_to_index: Mapping[str, int],
    *,
    num_artists: int = 4,
    clips_per_artist: int = 16,
    seed: int = 42,
    allowed_artist_keys: Sequence[str] | set[str] | None = None,
    minimum_token_frames: int | None = None,
) -> tuple[CohortItem, ...]:
    """Select a fixed balanced cohort without replacement."""

    if num_artists <= 0 or clips_per_artist <= 0:
        raise ValueError("num_artists and clips_per_artist must be positive")
    if minimum_token_frames is not None and minimum_token_frames <= 0:
        raise ValueError("minimum_token_frames must be positive when supplied")
    allowed = None if allowed_artist_keys is None else set(allowed_artist_keys)
    indices_by_artist: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        key = _record_artist_key(record)
        if key not in artist_to_index or (allowed is not None and key not in allowed):
            continue
        if (
            minimum_token_frames is not None
            and int(record.get("token_length", 0)) < minimum_token_frames
        ):
            continue
        indices_by_artist.setdefault(key, []).append(index)
    eligible = sorted(
        key
        for key, indices in indices_by_artist.items()
        if len(indices) >= clips_per_artist
    )
    if len(eligible) < num_artists:
        raise ValueError(
            f"need {num_artists} artists with at least {clips_per_artist} clips; "
            f"found {len(eligible)}"
        )

    rng = random.Random(int(seed))
    selected_artists = rng.sample(eligible, num_artists)
    items: list[CohortItem] = []
    for artist_key in selected_artists:
        selected_indices = rng.sample(indices_by_artist[artist_key], clips_per_artist)
        items.extend(
            CohortItem(
                dataset_index=index,
                artist_key=artist_key,
                artist_index=int(artist_to_index[artist_key]),
            )
            for index in selected_indices
        )
    return tuple(items)


def audio_frame_rate(model: Any) -> float:
    """Read the EnCodec token rate from a wrapped MusicGen model."""

    generator = getattr(model, "model", model)
    config = getattr(getattr(generator, "backbone", None), "config", None)
    audio_config = getattr(config, "audio_encoder", None)
    sampling_rate = float(getattr(audio_config, "sampling_rate", 32_000))
    ratios = getattr(audio_config, "upsampling_ratios", None)
    if ratios:
        return sampling_rate / math.prod(int(value) for value in ratios)
    return 50.0


def attach_audio_prompt(
    batch: Mapping[str, Any], *, prompt_frames: int
) -> dict[str, Any]:
    """Attach the real EnCodec prefix as MusicGen continuation inputs."""

    if prompt_frames <= 0:
        raise ValueError("prompt_frames must be positive")
    tokens = batch.get("audio_tokens")
    mask = batch.get("decoder_attention_mask")
    if not isinstance(tokens, Tensor) or tokens.ndim != 3:
        raise ValueError("audio_tokens must have shape [B,Q,T]")
    if not isinstance(mask, Tensor) or mask.shape != (tokens.shape[0], tokens.shape[2]):
        raise ValueError("decoder_attention_mask must have shape [B,T]")
    if bool((mask.long().sum(dim=-1) < prompt_frames).any()):
        raise ValueError("at least one source clip is shorter than the audio prompt")
    batch_size, codebooks, _ = tokens.shape
    result = dict(batch)
    result["generation_inputs"] = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "decoder_input_ids": tokens[..., :prompt_frames]
        .reshape(batch_size * codebooks, prompt_frames)
        .contiguous(),
    }
    return result


def fixed_audio_segment(
    audio: Tensor,
    *,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> Tensor:
    """Return a zero-padded ``[B,C,T]`` segment with an exact duration."""

    if audio.ndim == 2:
        audio = audio.unsqueeze(1)
    if audio.ndim != 3:
        raise ValueError("audio must have shape [B,T] or [B,C,T]")
    if sample_rate <= 0 or start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("invalid sample rate or segment duration")
    start = int(round(start_seconds * sample_rate))
    length = int(round(duration_seconds * sample_rate))
    segment = audio[..., start : start + length]
    if segment.shape[-1] == length:
        return segment
    padded = audio.new_zeros((*audio.shape[:-1], length))
    padded[..., : segment.shape[-1]] = segment
    return padded


def summarize_values(values: Tensor | Sequence[float]) -> MeanStd:
    """Return finite-value population mean/std for one metric."""

    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return MeanStd(float("nan"), float("nan"), 0)
    return MeanStd(
        mean=float(tensor.mean().item()),
        std=float(tensor.std(correction=0).item()),
        count=int(tensor.numel()),
    )


def format_mean_std(summary: MeanStd, *, precision: int = 4) -> str:
    if summary.count == 0:
        return "NaN ± NaN"
    return f"{summary.mean:.{precision}f} ± {summary.std:.{precision}f}"


def summarize_scenario_metrics(
    values: Mapping[str, Mapping[str, Tensor | Sequence[float]]],
) -> dict[str, dict[str, MeanStd]]:
    """Summarize all six-by-eight cells in stable display order."""

    summaries: dict[str, dict[str, MeanStd]] = {}
    for scenario in SCENARIO_NAMES:
        scenario_values = values.get(scenario, {})
        summaries[scenario] = {
            metric: summarize_values(scenario_values.get(metric, ()))
            for metric in EVALUATION_METRICS
        }
    return summaries


def formatted_table_rows(
    summaries: Mapping[str, Mapping[str, MeanStd]],
    *,
    precision: int = 4,
) -> list[list[str]]:
    """Create W&B-ready rows: six scenarios by eight ``mean ± std`` cells."""

    return [
        [
            SCENARIO_LABELS[scenario],
            *(
                format_mean_std(summaries[scenario][metric], precision=precision)
                for metric in EVALUATION_METRICS
            ),
        ]
        for scenario in SCENARIO_NAMES
    ]


__all__ = [
    "CohortItem",
    "EVALUATION_METRICS",
    "METRIC_LABELS",
    "MeanStd",
    "SCENARIO_LABELS",
    "SCENARIO_NAMES",
    "ScenarioCondition",
    "attach_audio_prompt",
    "audio_frame_rate",
    "build_control_scenarios",
    "fixed_audio_segment",
    "format_mean_std",
    "formatted_table_rows",
    "select_balanced_cohort",
    "summarize_scenario_metrics",
    "summarize_values",
]
