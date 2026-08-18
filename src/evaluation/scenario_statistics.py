"""Artist-level repeated-measures statistics for matched control scenarios."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math
import zlib
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class OmnibusTestResult:
    """One six-scenario omnibus result for a single evaluation metric."""

    metric: str
    count: int
    scenario_count: int
    f_statistic: float
    df_numerator: float
    df_denominator: float
    p_value: float
    gg_epsilon: float
    gg_df_numerator: float
    gg_df_denominator: float
    gg_p_value: float
    friedman_statistic: float
    friedman_p_value: float


@dataclass(frozen=True)
class PairwiseTestResult:
    """One paired post-hoc comparison between two control scenarios."""

    metric: str
    scenario_a: str
    scenario_b: str
    count: int
    mean_a: float
    mean_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    permutation_p_value: float
    holm_p_value: float = float("nan")
    significant: bool = False


def _as_finite_vector(values: Tensor | Sequence[float]) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float64).detach().cpu().flatten()


def _stable_artist_keys(artist_keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(key) for key in artist_keys))


def aggregate_metric_by_artist(
    values: Tensor | Sequence[float],
    artist_keys: Sequence[str],
) -> Tensor:
    """Return one value per artist without treating clips as independent units.

    Clip-level metrics are averaged within artist. Metrics that already contain
    exactly one value per artist (currently FAD) are returned unchanged. The
    latter relies on the callback's stable first-seen artist order.
    """

    tensor = _as_finite_vector(values)
    keys = tuple(str(key) for key in artist_keys)
    unique_keys = _stable_artist_keys(keys)
    if tensor.numel() == len(unique_keys):
        return tensor
    if tensor.numel() != len(keys):
        raise ValueError(
            "metric values must contain either one value per clip or one value "
            f"per artist; got {tensor.numel()} values for {len(keys)} clips and "
            f"{len(unique_keys)} artists"
        )
    return torch.stack(
        [
            tensor[
                torch.tensor([key == artist for key in keys], dtype=torch.bool)
            ].mean()
            for artist in unique_keys
        ]
    )


def artist_metric_matrix(
    values: Mapping[str, Mapping[str, Tensor | Sequence[float]]],
    *,
    metric: str,
    scenarios: Sequence[str],
    artist_keys: Sequence[str],
) -> Tensor:
    """Build a finite ``[artists, scenarios]`` repeated-measures matrix."""

    columns = []
    for scenario in scenarios:
        try:
            raw = values[scenario][metric]
        except KeyError as error:
            raise KeyError(
                f"missing {metric!r} values for scenario {scenario!r}"
            ) from error
        columns.append(aggregate_metric_by_artist(raw, artist_keys))
    lengths = {column.numel() for column in columns}
    if len(lengths) != 1:
        raise ValueError(f"artist counts differ between scenarios for {metric!r}")
    matrix = torch.stack(columns, dim=1)
    return matrix[torch.isfinite(matrix).all(dim=1)]


def _greenhouse_geisser_epsilon(matrix: Tensor) -> float:
    """Estimate Greenhouse--Geisser epsilon from a repeated-measures matrix."""

    subjects, conditions = matrix.shape
    if subjects < 2 or conditions < 2:
        return float("nan")
    covariance = torch.cov(matrix.T)
    centering = torch.eye(conditions, dtype=matrix.dtype) - torch.full(
        (conditions, conditions), 1.0 / conditions, dtype=matrix.dtype
    )
    centered_covariance = centering @ covariance @ centering
    trace = torch.trace(centered_covariance)
    denominator = (conditions - 1) * torch.trace(
        centered_covariance @ centered_covariance
    )
    if not torch.isfinite(denominator) or denominator <= 0:
        return 1.0
    epsilon = float((trace.square() / denominator).item())
    return min(1.0, max(1.0 / (conditions - 1), epsilon))


def repeated_measures_anova(matrix: Tensor, *, metric: str) -> OmnibusTestResult:
    """Run one-way repeated-measures ANOVA plus GG and Friedman checks."""

    from scipy.stats import f as f_distribution
    from scipy.stats import friedmanchisquare

    matrix = torch.as_tensor(matrix, dtype=torch.float64).detach().cpu()
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape [subjects, scenarios]")
    matrix = matrix[torch.isfinite(matrix).all(dim=1)]
    subjects, conditions = matrix.shape
    if subjects < 2:
        raise ValueError("repeated-measures tests need at least two artists")
    if conditions < 2:
        raise ValueError("repeated-measures tests need at least two scenarios")

    grand_mean = matrix.mean()
    condition_means = matrix.mean(dim=0)
    subject_means = matrix.mean(dim=1)
    ss_total = (matrix - grand_mean).square().sum()
    ss_condition = subjects * (condition_means - grand_mean).square().sum()
    ss_subject = conditions * (subject_means - grand_mean).square().sum()
    ss_error = torch.clamp(ss_total - ss_condition - ss_subject, min=0.0)
    df_numerator = float(conditions - 1)
    df_denominator = float((subjects - 1) * (conditions - 1))
    ms_condition = float((ss_condition / df_numerator).item())
    ms_error = float((ss_error / df_denominator).item())
    if ms_error == 0.0:
        f_statistic = 0.0 if ms_condition == 0.0 else float("inf")
    else:
        f_statistic = ms_condition / ms_error
    p_value = (
        0.0
        if math.isinf(f_statistic)
        else float(f_distribution.sf(f_statistic, df_numerator, df_denominator))
    )

    epsilon = _greenhouse_geisser_epsilon(matrix)
    gg_df_numerator = epsilon * df_numerator
    gg_df_denominator = epsilon * df_denominator
    gg_p_value = (
        0.0
        if math.isinf(f_statistic)
        else float(f_distribution.sf(f_statistic, gg_df_numerator, gg_df_denominator))
    )

    if bool((matrix == matrix[:, :1]).all()):
        friedman_statistic, friedman_p_value = 0.0, 1.0
    else:
        friedman = friedmanchisquare(
            *(matrix[:, index].numpy() for index in range(conditions))
        )
        friedman_statistic = float(friedman.statistic)
        friedman_p_value = float(friedman.pvalue)

    return OmnibusTestResult(
        metric=str(metric),
        count=int(subjects),
        scenario_count=int(conditions),
        f_statistic=float(f_statistic),
        df_numerator=df_numerator,
        df_denominator=df_denominator,
        p_value=p_value,
        gg_epsilon=epsilon,
        gg_df_numerator=gg_df_numerator,
        gg_df_denominator=gg_df_denominator,
        gg_p_value=gg_p_value,
        friedman_statistic=friedman_statistic,
        friedman_p_value=friedman_p_value,
    )


def paired_mean_permutation_test(
    scenario_a: Tensor,
    scenario_b: Tensor,
    *,
    num_resamples: int = 10_000,
    exact_max_pairs: int = 16,
    seed: int = 0,
) -> float:
    """Two-sided paired sign-flip test of the mean difference."""

    if num_resamples <= 0 or exact_max_pairs < 0:
        raise ValueError(
            "num_resamples must be positive and exact_max_pairs non-negative"
        )
    a = _as_finite_vector(scenario_a)
    b = _as_finite_vector(scenario_b)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same shape")
    finite = torch.isfinite(a) & torch.isfinite(b)
    differences = b[finite] - a[finite]
    count = differences.numel()
    if count == 0:
        return float("nan")
    observed = differences.mean().abs()
    tolerance = torch.finfo(differences.dtype).eps * max(1.0, float(observed)) * 16

    if count <= exact_max_pairs:
        permutations = 1 << count
        codes = torch.arange(permutations, dtype=torch.long).unsqueeze(1)
        bits = (codes >> torch.arange(count, dtype=torch.long)) & 1
        signs = bits.to(differences.dtype).mul_(2).sub_(1)
        null_statistics = (signs * differences).mean(dim=1).abs()
        return float((null_statistics >= observed - tolerance).double().mean().item())

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    extreme = 0
    completed = 0
    while completed < num_resamples:
        batch = min(4096, num_resamples - completed)
        signs = (
            torch.randint(
                0,
                2,
                (batch, count),
                generator=generator,
                dtype=torch.int8,
            )
            .to(differences.dtype)
            .mul_(2)
            .sub_(1)
        )
        null_statistics = (signs * differences).mean(dim=1).abs()
        extreme += int((null_statistics >= observed - tolerance).sum().item())
        completed += batch
    # The add-one correction prevents zero Monte Carlo p-values.
    return float((extreme + 1) / (num_resamples + 1))


def paired_mean_bootstrap_interval(
    scenario_a: Tensor,
    scenario_b: Tensor,
    *,
    num_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the artist-level paired mean delta."""

    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    a = _as_finite_vector(scenario_a)
    b = _as_finite_vector(scenario_b)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same shape")
    differences = (b - a)[torch.isfinite(a) & torch.isfinite(b)]
    if differences.numel() == 0:
        return float("nan"), float("nan")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        differences.numel(),
        (num_resamples, differences.numel()),
        generator=generator,
    )
    estimates = differences[indices].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    bounds = torch.quantile(
        estimates, torch.tensor([tail, 1.0 - tail], dtype=estimates.dtype)
    )
    return float(bounds[0].item()), float(bounds[1].item())


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise error correction."""

    adjusted = [float("nan")] * len(p_values)
    finite_indices = [
        index for index, value in enumerate(p_values) if math.isfinite(value)
    ]
    ordered = sorted(finite_indices, key=lambda index: float(p_values[index]))
    running = 0.0
    family_size = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = (family_size - rank) * float(p_values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def evaluate_scenario_statistics(
    values: Mapping[str, Mapping[str, Tensor | Sequence[float]]],
    *,
    metrics: Sequence[str],
    scenarios: Sequence[str],
    artist_keys: Sequence[str],
    num_permutations: int = 10_000,
    num_bootstrap_resamples: int = 2_000,
    confidence: float = 0.95,
    alpha: float = 0.05,
    exact_max_pairs: int = 16,
    seed: int = 0,
) -> tuple[list[OmnibusTestResult], list[PairwiseTestResult]]:
    """Evaluate all metrics and all scenario pairs at artist level."""

    if len(scenarios) < 2:
        raise ValueError("at least two scenarios are required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")

    omnibus_results: list[OmnibusTestResult] = []
    pairwise_results: list[PairwiseTestResult] = []
    for metric_index, metric in enumerate(metrics):
        matrix = artist_metric_matrix(
            values,
            metric=metric,
            scenarios=scenarios,
            artist_keys=artist_keys,
        )
        omnibus = repeated_measures_anova(matrix, metric=metric)
        omnibus_results.append(omnibus)

        metric_pairs: list[PairwiseTestResult] = []
        for pair_index, (index_a, index_b) in enumerate(
            combinations(range(len(scenarios)), 2)
        ):
            scenario_a = scenarios[index_a]
            scenario_b = scenarios[index_b]
            sample_a = matrix[:, index_a]
            sample_b = matrix[:, index_b]
            pair_seed = int(seed) + zlib.crc32(
                f"{metric_index}:{metric}:{pair_index}:{scenario_a}:{scenario_b}".encode()
            )
            ci_low, ci_high = paired_mean_bootstrap_interval(
                sample_a,
                sample_b,
                num_resamples=num_bootstrap_resamples,
                confidence=confidence,
                seed=pair_seed,
            )
            metric_pairs.append(
                PairwiseTestResult(
                    metric=metric,
                    scenario_a=scenario_a,
                    scenario_b=scenario_b,
                    count=int(matrix.shape[0]),
                    mean_a=float(sample_a.mean().item()),
                    mean_b=float(sample_b.mean().item()),
                    mean_difference=float((sample_b - sample_a).mean().item()),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    permutation_p_value=paired_mean_permutation_test(
                        sample_a,
                        sample_b,
                        num_resamples=num_permutations,
                        exact_max_pairs=exact_max_pairs,
                        seed=pair_seed,
                    ),
                )
            )

        adjusted = holm_adjust([result.permutation_p_value for result in metric_pairs])
        omnibus_significant = omnibus.gg_p_value < alpha
        pairwise_results.extend(
            replace(
                result,
                holm_p_value=corrected,
                significant=bool(omnibus_significant and corrected < alpha),
            )
            for result, corrected in zip(metric_pairs, adjusted, strict=True)
        )

    return omnibus_results, pairwise_results


__all__ = [
    "OmnibusTestResult",
    "PairwiseTestResult",
    "aggregate_metric_by_artist",
    "artist_metric_matrix",
    "evaluate_scenario_statistics",
    "holm_adjust",
    "paired_mean_bootstrap_interval",
    "paired_mean_permutation_test",
    "repeated_measures_anova",
]
