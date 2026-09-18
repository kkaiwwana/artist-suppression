"""Build the curated audio manifest used by the static project page.

The evaluation bundle is intentionally kept as the source of truth.  This
builder selects the six configurable variants for each example, copies them
into the project page, and precomputes compact waveform peaks so the browser
does not need to decode every WAV file on initial load.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import shutil
import sys
import wave
from typing import Any, Iterable


GROUP_CONFIG = {
    "demo single": {
        "id": "single",
        "label": "Single-target suppression",
        "target_key": "suppress_single_target",
        "others_key": "suppress_single_others",
    },
    "demo multiple": {
        "id": "multiple",
        "label": "Multiple-target suppression",
        "target_key": "suppress_multiple_target",
        "others_key": "suppress_multiple_others",
    },
    "suppress single": {
        "id": "single",
        "label": "Single-target suppression",
        "target_key": "suppress_single_target",
        "others_key": "suppress_single_others",
    },
    "suppress multiple": {
        "id": "multiple",
        "label": "Multiple-target suppression",
        "target_key": "suppress_multiple_target",
        "others_key": "suppress_multiple_others",
    },
}

BASE_VARIANTS = (
    ("prompt", "Prompt", "prompt"),
    ("reference", "Reference (GT)", "reference"),
    ("no_control", "No Control", "no_control"),
    ("enhance", "Enhance", "enhance_target"),
)

DISPLAY_ORDER_INSTRUCTIONS = (
    "Move complete sample lines to reorder; set show to false to hide one. "
    "New samples are appended when include_unlisted_samples is true. Suppress "
    "Others is controlled by the switch on the page."
)


def resolve_group(item: dict[str, Any]) -> dict[str, str]:
    """Return the demo group from either the legacy title or the bundle note."""

    for field in ("title", "note"):
        raw_value = str(item.get(field, "")).strip().lower()
        normalized = " ".join(raw_value.replace("_", " ").replace("-", " ").split())
        if normalized in GROUP_CONFIG:
            return GROUP_CONFIG[normalized]
    raise ValueError(
        "Item has no recognized single/multiple group in title or note: "
        f"title={item.get('title')!r}, note={item.get('note')!r}"
    )


def extract_waveform_peaks(audio_path: Path, points: int = 180) -> tuple[list[float], float]:
    """Return normalized absolute peaks and duration for a PCM-16 WAV file."""

    if points <= 0:
        raise ValueError("points must be a positive integer")

    with wave.open(str(audio_path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"Expected PCM-16 audio, got {wav_file.getsampwidth() * 8}-bit: {audio_path}")
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(frame_count)

    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()

    duration = frame_count / frame_rate if frame_rate else 0.0
    if not samples:
        return [0.0] * points, duration

    samples_per_bucket = max(1, len(samples) // points)
    peaks: list[float] = []
    for bucket_index in range(points):
        start = bucket_index * samples_per_bucket
        end = len(samples) if bucket_index == points - 1 else min(
            len(samples), (bucket_index + 1) * samples_per_bucket
        )
        if start >= len(samples):
            peaks.append(0.0)
            continue
        peak = max(abs(sample) for sample in samples[start:end])
        peaks.append(round(min(1.0, peak / 32768.0), 4))

    # Keep the channel read above explicit: it documents that interleaved
    # multi-channel samples are intentionally reduced into a single envelope.
    _ = channels
    return peaks, duration


def variant_specs(item: dict[str, Any]) -> list[dict[str, str]]:
    """Map one evaluation item to the six variants available to the web page."""

    group = resolve_group(item)
    variants = [
        {"id": variant_id, "label": label, "source_key": source_key}
        for variant_id, label, source_key in BASE_VARIANTS
    ]
    variants.extend(
        (
            {"id": "suppress", "label": "Suppress Target", "source_key": group["target_key"]},
            {
                "id": "suppress_others",
                "label": "Suppress Others",
                "source_key": group["others_key"],
            },
        )
    )
    return variants


def _resolve_inside(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Audio path escapes bundle root: {relative_path}") from exc
    return candidate


def _sample_slug(item: dict[str, Any], ordinal: int) -> str:
    digest = hashlib.sha1(str(item["sample_id"]).encode("utf-8")).hexdigest()[:8]
    return f"sample-{ordinal:02d}-{digest}"


def _serialize_display_order(config: dict[str, Any]) -> str:
    """Keep curated sample entries on one line so manual reordering stays easy."""

    lines = ["{"]
    keys = list(config)
    for key_index, key in enumerate(keys):
        suffix = "," if key_index < len(keys) - 1 else ""
        encoded_key = json.dumps(key, ensure_ascii=False)
        value = config[key]
        if key in ("single", "multiple") and isinstance(value, list):
            lines.append(f"  {encoded_key}: [")
            for entry_index, entry in enumerate(value):
                entry_suffix = "," if entry_index < len(value) - 1 else ""
                encoded_entry = json.dumps(entry, ensure_ascii=False)
                lines.append(f"    {encoded_entry}{entry_suffix}")
            lines.append(f"  ]{suffix}")
        else:
            encoded_value = json.dumps(value, ensure_ascii=False)
            lines.append(f"  {encoded_key}: {encoded_value}{suffix}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def sync_display_order(output_root: Path, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Synchronize demo curation without overwriting manual order or visibility."""

    output_root = output_root.resolve()
    config_path = output_root / "display_order.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {
            "_instructions": DISPLAY_ORDER_INSTRUCTIONS,
            "include_unlisted_samples": True,
            "single": [],
            "multiple": [],
        }

    config.pop("show_suppress_others", None)
    include_unlisted = config.get("include_unlisted_samples", True)
    if not isinstance(include_unlisted, bool):
        raise ValueError("display_order.json include_unlisted_samples must be true or false")

    item_list = list(items)
    for group in ("single", "multiple"):
        group_items = [item for item in item_list if item.get("group") == group]
        by_sample_id = {str(item["sample_id"]): item for item in group_items}
        by_sample_index = {
            str(item["sample_index"]): item
            for item in group_items
            if item.get("sample_index") is not None
        }
        configured_entries = config.get(group, [])
        if not isinstance(configured_entries, list):
            raise ValueError(f"display_order.json {group} must be a list")

        synced_entries: list[dict[str, Any]] = []
        seen_sample_ids: set[str] = set()
        for raw_entry in configured_entries:
            if isinstance(raw_entry, dict):
                entry = dict(raw_entry)
                sample_id = entry.get("sample_id")
                sample_index = entry.get("sample_index")
            else:
                entry = {"sample_index": raw_entry}
                sample_id = None
                sample_index = raw_entry

            item = None
            if sample_id is not None:
                item = by_sample_id.get(str(sample_id))
            if item is None and sample_index is not None:
                item = by_sample_index.get(str(sample_index))
            if item is None or str(item["sample_id"]) in seen_sample_ids:
                continue

            show = entry.get("show", True)
            if not isinstance(show, bool):
                raise ValueError(
                    f"display_order.json show must be true or false for {item['sample_id']}"
                )
            if item.get("sample_index") is not None:
                entry["sample_index"] = item["sample_index"]
            else:
                entry["sample_id"] = item["sample_id"]
            entry["show"] = show
            entry["title"] = item.get("source_title") or "Untitled source"
            synced_entries.append(entry)
            seen_sample_ids.add(str(item["sample_id"]))

        if include_unlisted:
            for item in group_items:
                sample_id = str(item["sample_id"])
                if sample_id in seen_sample_ids:
                    continue
                identifier = (
                    {"sample_index": item["sample_index"]}
                    if item.get("sample_index") is not None
                    else {"sample_id": item["sample_id"]}
                )
                synced_entries.append(
                    {
                        **identifier,
                        "show": True,
                        "title": item.get("source_title") or "Untitled source",
                    }
                )
                seen_sample_ids.add(sample_id)

        config[group] = synced_entries

    config["_instructions"] = DISPLAY_ORDER_INSTRUCTIONS
    config["include_unlisted_samples"] = include_unlisted
    config_path.write_text(_serialize_display_order(config), encoding="utf-8")
    return config


def build_site_bundle(
    source_manifest: Path,
    output_root: Path,
    *,
    peak_points: int = 180,
    excluded_sample_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Create the web manifest and copy its curated audio files."""

    source_manifest = source_manifest.resolve()
    source_root = source_manifest.parent
    output_root = output_root.resolve()
    audio_output = output_root / "audio"
    audio_output.mkdir(parents=True, exist_ok=True)

    source_data = json.loads(source_manifest.read_text(encoding="utf-8"))
    excluded = set(excluded_sample_ids)
    output_items: list[dict[str, Any]] = []

    for ordinal, item in enumerate(source_data.get("items", []), start=1):
        if item.get("sample_id") in excluded:
            continue
        group = resolve_group(item)
        slug = _sample_slug(item, ordinal)
        variants: list[dict[str, Any]] = []

        for spec in variant_specs(item):
            source_relative = item.get("audio", {}).get(spec["source_key"])
            if not source_relative:
                raise ValueError(
                    f"Item {item.get('sample_id')} is missing audio field {spec['source_key']}"
                )
            source_audio = _resolve_inside(source_root, str(source_relative))
            if not source_audio.is_file():
                raise FileNotFoundError(source_audio)

            destination_relative = Path("audio") / slug / f"{spec['id']}.wav"
            destination_audio = output_root / destination_relative
            destination_audio.parent.mkdir(parents=True, exist_ok=True)
            if (
                not destination_audio.exists()
                or destination_audio.stat().st_size != source_audio.stat().st_size
            ):
                shutil.copy2(source_audio, destination_audio)

            peaks, duration = extract_waveform_peaks(source_audio, points=peak_points)
            variants.append(
                {
                    "id": spec["id"],
                    "label": spec["label"],
                    "source_key": spec["source_key"],
                    "src": destination_relative.as_posix(),
                    "duration_seconds": round(duration, 3),
                    "peaks": peaks,
                }
            )

        output_items.append(
            {
                "ordinal": ordinal,
                "sample_id": item["sample_id"],
                "sample_index": item.get("sample_index"),
                "group": group["id"],
                "group_label": group["label"],
                "artist": item.get("source_artist_name") or "Unknown artist",
                "source_title": item.get("source_title") or "Untitled source",
                "caption": item.get("caption") or "",
                "clip_id": item.get("clip_id"),
                "start_ms": item.get("start_ms"),
                "end_ms": item.get("end_ms"),
                "selection_evidence": item.get("selection_evidence"),
                "variants": variants,
            }
        )

    output_manifest = {
        "schema_version": 1,
        "source_schema_version": source_data.get("schema_version"),
        "checkpoint_id": source_data.get("checkpoint_id"),
        "generation_id": source_data.get("generation_id"),
        "excluded_sample_ids": sorted(excluded),
        "group_order": ["single", "multiple"],
        "variant_order": [spec["id"] for spec in variant_specs(source_data["items"][0])],
        "items": output_items,
    }
    sync_display_order(output_root, output_items)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def validate_site_bundle(manifest_path: Path) -> dict[str, int]:
    """Validate the generated web bundle and return a compact summary."""

    manifest_path = manifest_path.resolve()
    output_root = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {"single": 0, "multiple": 0, "items": 0, "audio_files": 0}

    for item in data.get("items", []):
        group = item.get("group")
        if group not in ("single", "multiple"):
            raise ValueError(f"Unexpected generated group: {group!r}")
        counts[group] += 1
        counts["items"] += 1
        variants = item.get("variants", [])
        expected_variant_order = data.get("variant_order", [])
        actual_variant_order = [variant.get("id") for variant in variants]
        if actual_variant_order != expected_variant_order:
            raise ValueError(
                f"{item.get('sample_id')} has variant order {actual_variant_order}, "
                f"expected {expected_variant_order}"
            )
        for variant in variants:
            audio_path = _resolve_inside(output_root, variant["src"])
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            if not variant.get("peaks"):
                raise ValueError(f"No waveform peaks for {audio_path}")
            counts["audio_files"] += 1
    return counts


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=repo_root / "evaluation_outputs" / "paper_demo_bundle" / "manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "project_page" / "demo_data",
    )
    parser.add_argument("--peak-points", type=int, default=180)
    parser.add_argument(
        "--exclude-sample-id",
        action="append",
        default=[],
        help="Exclude one sample id. Existing exclusions in the output manifest are preserved.",
    )
    parser.add_argument(
        "--clear-exclusions",
        action="store_true",
        help="Discard exclusions stored in the existing output manifest before rebuilding.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing generated bundle without rebuilding it.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    manifest_path = args.output_root / "manifest.json"
    if args.check:
        summary = validate_site_bundle(manifest_path)
    else:
        existing_exclusions: set[str] = set()
        if manifest_path.is_file() and not args.clear_exclusions:
            existing_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_exclusions.update(existing_data.get("excluded_sample_ids", []))
        existing_exclusions.update(args.exclude_sample_id)
        build_site_bundle(
            args.source_manifest,
            args.output_root,
            peak_points=args.peak_points,
            excluded_sample_ids=existing_exclusions,
        )
        summary = validate_site_bundle(manifest_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
