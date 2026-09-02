"""Metric-driven browsing and portable paper-demo export.

The browser operates on the persisted evaluation bundle rather than model
objects.  Metric discovery is schema based, so additional per-clip or
per-artist CSVs appear without changing notebook code.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import soundfile as sf

from src.evaluation.control_distribution_analysis import (
    ATTRIBUTION_ARTIST_VALUES_NAME,
    ATTRIBUTION_VALUES_NAME,
    CLIP_MANIFEST_NAME,
    FAD_VALUES_NAME,
    KL_VALUES_NAME,
    RUN_METADATA_NAME,
)
from src.evaluation.control_scenarios import SCENARIO_LABELS, SCENARIO_NAMES


SELECTION_SCHEMA_VERSION = 1
COMPARISON_MODES = (
    "scenario_value",
    "delta_vs_no_control",
    "absolute_delta_vs_no_control",
    "degradation_vs_no_control",
    "improvement_vs_no_control",
)


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    label: str
    csv_path: str
    value_column: str
    entity_level: str
    entity_key: str
    higher_is_better: bool | None
    description: str = ""


_KNOWN_METRICS = (
    MetricSpec(
        "passt_kl_to_reference",
        "PaSST KL(reference || generated)",
        f"metrics/{KL_VALUES_NAME}",
        "kl_reference_to_generated",
        "clip",
        "sample_id",
        False,
        "One paired KL value per clip; lower means closer to reference.",
    ),
    MetricSpec(
        "fad_to_reference",
        "FAD to reference",
        f"metrics/{FAD_VALUES_NAME}",
        "fad",
        "artist",
        "artist_key",
        False,
        "Canonical group-to-group FAD: exactly one value per artist.",
    ),
    MetricSpec(
        "target_attribution_hit",
        "Target artist top-1 hit",
        f"metrics/{ATTRIBUTION_VALUES_NAME}",
        "target_attribution_hit",
        "clip",
        "sample_id",
        True,
        "A per-clip 0/1 hit. Its mean, not an individual value, is attribution rate.",
    ),
    MetricSpec(
        "target_artist_confidence",
        "Target artist confidence",
        f"metrics/{ATTRIBUTION_VALUES_NAME}",
        "target_artist_confidence",
        "clip",
        "sample_id",
        True,
        "Classifier probability assigned to the source/target artist.",
    ),
    MetricSpec(
        "target_artist_rank",
        "Target artist rank",
        f"metrics/{ATTRIBUTION_VALUES_NAME}",
        "target_artist_rank",
        "clip",
        "sample_id",
        False,
        "One-based classifier rank of the source/target artist.",
    ),
    MetricSpec(
        "target_attribution_rate",
        "Target artist attribution rate",
        f"metrics/{ATTRIBUTION_ARTIST_VALUES_NAME}",
        "target_attribution_rate",
        "artist",
        "artist_key",
        True,
        "Mean target-artist top-1 hit over all clips of one artist.",
    ),
    MetricSpec(
        "mean_target_artist_confidence",
        "Mean target artist confidence",
        f"metrics/{ATTRIBUTION_ARTIST_VALUES_NAME}",
        "mean_target_artist_confidence",
        "artist",
        "artist_key",
        True,
        "Mean target-artist classifier confidence over one artist group.",
    ),
    MetricSpec(
        "mean_target_artist_rank",
        "Mean target artist rank",
        f"metrics/{ATTRIBUTION_ARTIST_VALUES_NAME}",
        "mean_target_artist_rank",
        "artist",
        "artist_key",
        False,
        "Mean one-based target-artist rank over one artist group.",
    ),
)


_NON_METRIC_COLUMNS = {
    "scenario",
    "scenario_label",
    "sample_index",
    "sample_id",
    "checkpoint_id",
    "artist_key",
    "artist_index",
    "artist_clip_index",
    "track_id",
    "clip_id",
    "start_ms",
    "end_ms",
    "caption_sha256_16",
    "predicted_artist_index",
    "predicted_artist_key",
    "num_clips",
    "num_reference_windows",
    "num_generated_windows",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _numeric_columns(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return []
    output: list[str] = []
    for column in rows[0]:
        if column in _NON_METRIC_COLUMNS:
            continue
        values = [row.get(column, "") for row in rows]
        nonempty = [value for value in values if str(value).strip()]
        if not nonempty:
            continue
        try:
            for value in nonempty:
                float(value)
        except (TypeError, ValueError):
            continue
        output.append(column)
    return output


def discover_metric_specs(output_dir: str | Path) -> dict[str, MetricSpec]:
    """Discover built-in and arbitrary long-form scenario metric CSVs."""

    root = Path(output_dir).expanduser().resolve()
    metrics: dict[str, MetricSpec] = {}
    claimed: set[tuple[str, str]] = set()
    for spec in _KNOWN_METRICS:
        path = root / spec.csv_path
        if path.is_file():
            header = _read_csv(path)
            if header and spec.value_column in header[0]:
                metrics[spec.metric_id] = spec
                claimed.add((spec.csv_path, spec.value_column))

    metrics_dir = root / "metrics"
    if not metrics_dir.is_dir():
        return metrics
    for path in sorted(metrics_dir.glob("*.csv")):
        rows = _read_csv(path)
        if not rows or "scenario" not in rows[0]:
            continue
        if "sample_id" in rows[0]:
            level, key = "clip", "sample_id"
        elif "sample_index" in rows[0]:
            level, key = "clip", "sample_index"
        elif "artist_key" in rows[0]:
            level, key = "artist", "artist_key"
        else:
            continue
        relative = path.relative_to(root).as_posix()
        for column in _numeric_columns(rows):
            if (relative, column) in claimed:
                continue
            metric_id = f"{path.stem}:{column}"
            metrics[metric_id] = MetricSpec(
                metric_id=metric_id,
                label=column.replace("_", " ").title(),
                csv_path=relative,
                value_column=column,
                entity_level=level,
                entity_key=key,
                higher_is_better=None,
                description="Automatically discovered; choose sorting semantics explicitly.",
            )
    return metrics


def _mean_by_key(
    rows: Sequence[Mapping[str, str]], *, key: str, value: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row[value]))
    return {
        group_key: sum(group_values) / len(group_values)
        for group_key, group_values in grouped.items()
    }


def rank_metric_candidates(
    output_dir: str | Path,
    metric_id: str,
    *,
    scenario: str,
    comparison_mode: str = "absolute_delta_vs_no_control",
    largest_first: bool = True,
    top_k: int = 20,
    artist_key: str | None = None,
    caption_query: str | None = None,
) -> list[dict[str, Any]]:
    """Rank clip or artist entities for one metric/scenario comparison."""

    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario: {scenario}")
    if comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"unknown comparison_mode: {comparison_mode}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    root = Path(output_dir).expanduser().resolve()
    specs = discover_metric_specs(root)
    if metric_id not in specs:
        raise KeyError(f"metric is unavailable: {metric_id}")
    spec = specs[metric_id]
    metric_rows = _read_csv(root / spec.csv_path)
    scenario_rows = [row for row in metric_rows if row["scenario"] == scenario]
    if not scenario_rows:
        return []
    current = _mean_by_key(scenario_rows, key=spec.entity_key, value=spec.value_column)
    baseline = _mean_by_key(
        [row for row in metric_rows if row["scenario"] == "no_control"],
        key=spec.entity_key,
        value=spec.value_column,
    )
    clips = _read_csv(root / CLIP_MANIFEST_NAME)
    clips_by_sample_id = {
        row.get("sample_id", row["sample_index"]): row for row in clips
    }
    clips_by_index = {row["sample_index"]: row for row in clips}
    clips_by_artist: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clips:
        clips_by_artist[row["artist_key"]].append(row)

    def source_row(entity_id: str) -> dict[str, str] | None:
        if spec.entity_level == "artist":
            values = clips_by_artist.get(entity_id, [])
            return values[0] if values else None
        return clips_by_sample_id.get(entity_id) or clips_by_index.get(entity_id)

    candidates: list[dict[str, Any]] = []
    query = (caption_query or "").casefold().strip()
    for entity_id, value in current.items():
        clip = source_row(entity_id)
        if clip is None:
            continue
        if artist_key and clip["artist_key"] != artist_key:
            continue
        if query:
            texts = (
                [row["text"] for row in clips_by_artist[clip["artist_key"]]]
                if spec.entity_level == "artist"
                else [clip["text"]]
            )
            if not any(query in text.casefold() for text in texts):
                continue
        baseline_value = baseline.get(entity_id)
        delta = None if baseline_value is None else value - baseline_value
        if comparison_mode == "scenario_value":
            score = value
        elif delta is None:
            continue
        elif comparison_mode == "delta_vs_no_control":
            score = delta
        elif comparison_mode == "absolute_delta_vs_no_control":
            score = abs(delta)
        elif comparison_mode == "degradation_vs_no_control":
            if spec.higher_is_better is None:
                raise ValueError(
                    "degradation mode needs a known higher_is_better direction"
                )
            score = -delta if spec.higher_is_better else delta
        else:
            if spec.higher_is_better is None:
                raise ValueError(
                    "improvement mode needs a known higher_is_better direction"
                )
            score = delta if spec.higher_is_better else -delta
        candidates.append(
            {
                "metric_id": spec.metric_id,
                "metric_label": spec.label,
                "entity_level": spec.entity_level,
                "entity_id": entity_id,
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "comparison_mode": comparison_mode,
                "scenario_value": value,
                "no_control_value": baseline_value,
                "delta_vs_no_control": delta,
                "ranking_score": float(score),
                "higher_is_better": spec.higher_is_better,
                "artist_key": clip["artist_key"],
                "artist_index": int(clip["artist_index"]),
                "track_id": (
                    clip.get("track_id") if spec.entity_level == "clip" else None
                ),
                "sample_id": (
                    clip.get("sample_id") if spec.entity_level == "clip" else None
                ),
                "text": clip.get("text") if spec.entity_level == "clip" else None,
                "num_clips": (
                    len(clips_by_artist[clip["artist_key"]])
                    if spec.entity_level == "artist"
                    else 1
                ),
            }
        )
    candidates.sort(key=lambda row: row["ranking_score"], reverse=largest_first)
    for rank, row in enumerate(candidates[:top_k], start=1):
        row["rank"] = rank
    return candidates[:top_k]


def candidate_audio_clips(
    output_dir: str | Path,
    candidate: Mapping[str, Any],
    *,
    secondary_metric_id: str | None = None,
    secondary_comparison_mode: str = "absolute_delta_vs_no_control",
) -> list[dict[str, Any]]:
    """Resolve a candidate into matched clips with six scenario audio paths."""

    root = Path(output_dir).expanduser().resolve()
    clips = _read_csv(root / CLIP_MANIFEST_NAME)
    if candidate["entity_level"] == "clip":
        selected = [
            row for row in clips if row.get("sample_id") == candidate["entity_id"]
        ]
        if not selected:
            selected = [
                row for row in clips if row["sample_index"] == candidate["entity_id"]
            ]
    else:
        selected = [row for row in clips if row["artist_key"] == candidate["entity_id"]]
    selected.sort(key=lambda row: int(row["artist_clip_index"]))

    secondary_by_sample: dict[str, dict[str, Any]] = {}
    if candidate["entity_level"] == "artist" and secondary_metric_id:
        specs = discover_metric_specs(root)
        secondary = specs.get(secondary_metric_id)
        if secondary is None or secondary.entity_level != "clip":
            raise ValueError("FAD/artist candidates need a clip-level secondary metric")
        ranked = rank_metric_candidates(
            root,
            secondary_metric_id,
            scenario=str(candidate["scenario"]),
            comparison_mode=secondary_comparison_mode,
            largest_first=True,
            top_k=len(clips),
            artist_key=str(candidate["artist_key"]),
        )
        secondary_by_sample = {
            str(row["sample_id"]): row for row in ranked if row.get("sample_id")
        }
        order = {str(row["sample_id"]): index for index, row in enumerate(ranked)}
        selected.sort(key=lambda row: order.get(str(row.get("sample_id")), len(order)))

    output: list[dict[str, Any]] = []
    for row in selected:
        item = dict(row)
        item["reference_absolute_path"] = str((root / row["reference_path"]).resolve())
        item["prompt_absolute_path"] = (
            str((root / row["prompt_path"]).resolve())
            if row.get("prompt_path")
            else None
        )
        item["scenario_absolute_paths"] = {
            scenario: str((root / row[f"{scenario}_path"]).resolve())
            for scenario in SCENARIO_NAMES
        }
        item["secondary_selection_evidence"] = secondary_by_sample.get(
            str(row.get("sample_id"))
        )
        output.append(item)
    return output


def make_demo_selection(
    candidate: Mapping[str, Any],
    clip: Mapping[str, Any],
    *,
    title: str = "",
    note: str = "",
) -> dict[str, Any]:
    """Create a traceable selection record for one matched six-scenario clip."""

    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "sample_id": clip["sample_id"],
        "sample_index": int(clip["sample_index"]),
        "artist_key": clip["artist_key"],
        "track_id": clip.get("track_id", ""),
        "caption": clip.get("text", ""),
        "title": str(title),
        "note": str(note),
        "selection_evidence": {
            key: candidate.get(key)
            for key in (
                "metric_id",
                "metric_label",
                "entity_level",
                "entity_id",
                "scenario",
                "comparison_mode",
                "scenario_value",
                "no_control_value",
                "delta_vs_no_control",
                "ranking_score",
                "rank",
            )
        },
        "secondary_selection_evidence": clip.get("secondary_selection_evidence"),
    }


def load_selection_draft(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return []
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("selection draft must contain a JSON list")
    return [dict(item) for item in value]


def save_selection_draft(
    path: str | Path, selections: Sequence[Mapping[str, Any]]
) -> Path:
    target = Path(path).expanduser().resolve()
    _write_json(target, [dict(item) for item in selections])
    return target


def _convert_or_copy_audio(source: Path, target: Path, subtype: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if subtype is None:
        shutil.copy2(source, temporary)
    else:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        sf.write(temporary, audio, sample_rate, format="WAV", subtype=subtype)
    temporary.replace(target)


def export_paper_demo_bundle(
    output_dir: str | Path,
    selections: Sequence[Mapping[str, Any]],
    destination: str | Path,
    *,
    web_audio_subtype: str | None = "PCM_16",
) -> Path:
    """Create a self-contained, movable web/paper audio selection bundle."""

    if not selections:
        raise ValueError("at least one selection is required")
    root = Path(output_dir).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    clips = _read_csv(root / CLIP_MANIFEST_NAME)
    by_id = {row["sample_id"]: row for row in clips}
    run_metadata = json.loads((root / RUN_METADATA_NAME).read_text(encoding="utf-8"))
    exported: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for index, selection in enumerate(selections, start=1):
        sample_id = str(selection["sample_id"])
        if sample_id not in by_id:
            raise KeyError(f"selection sample_id is absent from clips.csv: {sample_id}")
        clip = by_id[sample_id]
        short_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
        item_dir = destination / "audio" / f"item_{index:02d}__{short_id}"
        paths: dict[str, str | None] = {}
        asset_fields = {
            "reference": clip["reference_path"],
            "prompt": clip.get("prompt_path", ""),
            **{scenario: clip[f"{scenario}_path"] for scenario in SCENARIO_NAMES},
        }
        for asset_name, relative_source in asset_fields.items():
            if not relative_source:
                paths[asset_name] = None
                continue
            source = (root / relative_source).resolve()
            target = item_dir / source.name
            _convert_or_copy_audio(source, target, web_audio_subtype)
            paths[asset_name] = target.relative_to(destination).as_posix()
            flat_rows.append(
                {
                    "selection_index": index,
                    "sample_id": sample_id,
                    "artist_key": clip["artist_key"],
                    "track_id": clip.get("track_id", ""),
                    "asset": asset_name,
                    "audio_path": paths[asset_name],
                }
            )
        exported.append(
            {
                **dict(selection),
                "checkpoint_id": run_metadata.get("checkpoint_id"),
                "generation_id": run_metadata.get("generation_id"),
                "artist_index": int(clip["artist_index"]),
                "clip_id": clip.get("clip_id", ""),
                "source_audio": clip.get("source_audio", ""),
                "source_artist_id": clip.get("source_artist_id", ""),
                "source_artist_name": clip.get("source_artist_name", ""),
                "source_title": clip.get("source_title", ""),
                "start_ms": int(clip["start_ms"]) if clip.get("start_ms") else None,
                "end_ms": int(clip["end_ms"]) if clip.get("end_ms") else None,
                "scenario_labels": dict(SCENARIO_LABELS),
                "audio": paths,
            }
        )
    manifest = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "checkpoint_id": run_metadata.get("checkpoint_id"),
        "generation_id": run_metadata.get("generation_id"),
        "path_base": "bundle_root",
        "web_audio_subtype": web_audio_subtype or "source",
        "scenario_order": list(SCENARIO_NAMES),
        "items": exported,
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_csv(destination / "audio_assets.csv", flat_rows)
    return manifest_path


def metric_specs_as_rows(specs: Mapping[str, MetricSpec]) -> list[dict[str, Any]]:
    return [asdict(spec) for spec in specs.values()]


__all__ = [
    "COMPARISON_MODES",
    "MetricSpec",
    "candidate_audio_clips",
    "discover_metric_specs",
    "export_paper_demo_bundle",
    "load_selection_draft",
    "make_demo_selection",
    "metric_specs_as_rows",
    "rank_metric_candidates",
    "save_selection_draft",
]
