"""Offline six-scenario audio export and KL/FAD distribution analysis.

The epoch callback intentionally keeps evaluation artifacts in memory because it
logs a single compact result.  This module serves a different use case: generate
an entire fixed cohort once, persist every waveform, and evaluate/rank it later
without keeping MusicGen and the external metric networks resident together.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch import Tensor

from src.callback.epoch_control_evaluation import (
    EpochControlEvaluationCallback,
    _move_to_device,
    _passt_per_sample,
    _vggish_embeddings,
)
from src.evaluation.checkpoint_runtime import restore_control_runtime
from src.evaluation.control_scenarios import (
    SCENARIO_LABELS,
    SCENARIO_NAMES,
    attach_audio_prompt,
    audio_frame_rate,
    fixed_audio_segment,
)
from src.evaluation.metric_runtimes import frechet_distance_from_embeddings


FORMAT_VERSION = 2
CLIP_MANIFEST_NAME = "clips.csv"
RUN_METADATA_NAME = "run_metadata.json"
GENERATED_MANIFEST_NAME = "generated_samples.jsonl"
KL_VALUES_NAME = "passt_kl_per_clip.csv"
FAD_VALUES_NAME = "fad_per_artist.csv"
ATTRIBUTION_VALUES_NAME = "artist_attribution_per_clip.csv"
ATTRIBUTION_ARTIST_VALUES_NAME = "artist_attribution_per_artist.csv"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _slug(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._-")
    return cleaned[:80] or fallback


def _short_slug(value: str, *, fallback: str, limit: int) -> str:
    return _slug(value, fallback=fallback)[: int(limit)]


def _record_identifiers(
    record: Mapping[str, Any], *, dataset_index: int
) -> dict[str, Any]:
    """Return stable source identifiers, including a caption disambiguator."""

    track_value = record.get("track_id")
    audio_value = str(record.get("audio", ""))
    if track_value is None or not str(track_value).strip():
        audio_parent = Path(audio_value).parent.name if audio_value else ""
        track_value = audio_parent.removeprefix("track_") or f"row_{dataset_index}"
    clip_value = record.get("clip_id")
    if clip_value is None or not str(clip_value).strip():
        clip_value = Path(audio_value).stem if audio_value else f"row_{dataset_index}"
    caption = unicodedata.normalize(
        "NFC", str(record.get("text", record.get("caption", "")))
    )
    caption = " ".join(caption.split())
    caption_hash = hashlib.sha256(caption.encode("utf-8")).hexdigest()[:16]

    def milliseconds(key: str) -> int | None:
        value = record.get(key)
        if value is None or str(value).strip() == "":
            return None
        try:
            return int(round(float(value) * 1000.0))
        except (TypeError, ValueError):
            return None

    start_ms = milliseconds("start_time")
    end_ms = milliseconds("end_time")
    match = re.search(r"_(\d{9})_(\d{9})$", Path(audio_value).stem)
    if match is not None:
        start_ms = int(match.group(1)) if start_ms is None else start_ms
        end_ms = int(match.group(2)) if end_ms is None else end_ms
    return {
        "track_id": str(track_value),
        "clip_id": str(clip_value),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "caption": caption,
        "caption_sha256_16": caption_hash,
    }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _absolute(root: Path, relative: str) -> Path:
    return (root / Path(relative)).resolve()


def _write_audio(path: Path, waveform: Tensor, sample_rate: int, subtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = torch.as_tensor(waveform).detach().float().cpu()
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = audio[0]
        else:
            audio = audio.transpose(0, 1)
    elif audio.ndim != 1:
        raise ValueError(
            f"one audio item must have shape [T] or [C,T], got {audio.shape}"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    sf.write(
        temporary,
        audio.numpy(),
        int(sample_rate),
        format="WAV",
        subtype=subtype,
    )
    info = sf.info(temporary)
    expected_frames = int(audio.shape[0])
    if info.samplerate != int(sample_rate) or info.frames != expected_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"audio verification failed for {path}: rate={info.samplerate}, "
            f"frames={info.frames}, expected={sample_rate}/{expected_frames}"
        )
    temporary.replace(path)


def _read_audio(path: Path, expected_sample_rate: int | None = None) -> Tensor:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(
            f"sample-rate mismatch for {path}: {sample_rate} != {expected_sample_rate}"
        )
    # soundfile is time-major; metrics accept [C,T] and are evaluated as mono.
    return torch.from_numpy(values.T.copy())


def _stack_audio(
    paths: Sequence[Path], *, sample_rate: int, expected_frames: int | None = None
) -> Tensor:
    clips = [_read_audio(path, sample_rate) for path in paths]
    if not clips:
        raise ValueError("cannot stack an empty audio list")
    frames = expected_frames or max(clip.shape[-1] for clip in clips)
    result = clips[0].new_zeros((len(clips), 1, frames))
    for index, clip in enumerate(clips):
        mono = clip.mean(dim=0, keepdim=True)
        length = min(frames, mono.shape[-1])
        result[index, :, :length] = mono[:, :length]
    return result


def _clip_identity_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Columns shared by every per-clip metric table."""

    return {
        "sample_index": int(row["sample_index"]),
        "sample_id": row.get("sample_id", f"sample_{int(row['sample_index']):04d}"),
        "checkpoint_id": row.get("checkpoint_id", ""),
        "artist_key": row["artist_key"],
        "artist_index": int(row["artist_index"]),
        "artist_clip_index": int(row["artist_clip_index"]),
        "track_id": row.get("track_id", ""),
        "clip_id": row.get("clip_id", ""),
        "start_ms": row.get("start_ms", ""),
        "end_ms": row.get("end_ms", ""),
        "caption_sha256_16": row.get("caption_sha256_16", ""),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty JSONL manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(dict(row), ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _asset_fingerprint(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    assets: list[tuple[str, int, int, int, str]] = []
    for row in rows:
        keys = ["reference_path", *(f"{name}_path" for name in SCENARIO_NAMES)]
        if row.get("prompt_path"):
            keys.append("prompt_path")
        for key in keys:
            relative = str(row[key])
            path = _absolute(root, relative)
            if not path.is_file():
                raise FileNotFoundError(f"generated audio asset is missing: {path}")
            stat = path.stat()
            info = sf.info(path)
            assets.append(
                (
                    relative,
                    stat.st_size,
                    int(info.samplerate),
                    int(info.frames),
                    _file_sha256(path),
                )
            )
    return _canonical_sha256(assets)


def _complete_generation(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata = json.loads((root / RUN_METADATA_NAME).read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError(f"generation is not complete under {root}")
    rows = _read_csv(root / CLIP_MANIFEST_NAME)
    observed = _asset_fingerprint(root, rows)
    expected = metadata.get("audio_assets_fingerprint")
    if expected != observed:
        raise RuntimeError(
            "audio assets changed after generation; repair/regenerate the bundle "
            "before computing metrics"
        )
    return metadata, rows


def _metric_cache(
    target: Path,
    *,
    generation_id: str,
    audio_assets_fingerprint: str,
    runtime_id: str,
    overwrite: bool,
) -> bool:
    sidecar = target.with_suffix(target.suffix + ".metadata.json")
    if not target.is_file() or overwrite:
        return False
    if not sidecar.is_file():
        raise RuntimeError(f"metric cache has no provenance sidecar: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("generation_id") != generation_id:
        raise RuntimeError("metric CSV belongs to a different generated audio bundle")
    if metadata.get("audio_assets_fingerprint") != audio_assets_fingerprint:
        raise RuntimeError(
            "generated audio changed since this metric CSV; pass overwrite=True"
        )
    if metadata.get("runtime_id") != runtime_id:
        raise RuntimeError(
            "metric CSV was produced by another runtime; pass overwrite=True"
        )
    return True


def _write_metric_sidecar(
    target: Path,
    *,
    generation_id: str,
    audio_assets_fingerprint: str,
    runtime_id: str,
    row_count: int,
) -> None:
    _json_dump(
        target.with_suffix(target.suffix + ".metadata.json"),
        {
            "format_version": FORMAT_VERSION,
            "generation_id": generation_id,
            "audio_assets_fingerprint": audio_assets_fingerprint,
            "runtime_id": runtime_id,
            "row_count": int(row_count),
        },
    )


def _callback_kwargs(
    config: Any,
    *,
    num_artists: int | None,
    clips_per_artist: int | None,
    generation_batch_size: int | None,
    generation_seed: int | None,
    cohort_seed: int | None,
    generation_mode: str | None,
    audio_prompt_seconds: float | None,
    continuation_seconds: float | None,
    respect_classifier_vocabulary: bool,
) -> dict[str, Any]:
    configured = config.runner.get("epoch_control_evaluation", {})
    values = OmegaConf.to_container(configured, resolve=True) if configured else {}
    values = dict(values or {})
    accepted = set(
        inspect.signature(EpochControlEvaluationCallback.__init__).parameters
    )
    values = {key: value for key, value in values.items() if key in accepted}
    values.pop("enabled", None)
    values.pop("runtime_factories", None)
    values.pop("evaluation_runner", None)
    values["qualitative_enabled"] = False
    if not respect_classifier_vocabulary:
        values["classifier_checkpoint_path"] = None
    overrides = {
        "num_artists": num_artists,
        "clips_per_artist": clips_per_artist,
        "generation_batch_size": generation_batch_size,
        "generation_seed": generation_seed,
        "cohort_seed": cohort_seed,
        "generation_mode": generation_mode,
        "audio_prompt_seconds": audio_prompt_seconds,
        "continuation_seconds": continuation_seconds,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return values


def _clip_rows(
    callback: EpochControlEvaluationCallback,
    dataset: Any,
    records: Sequence[Mapping[str, Any]],
    artist_to_index: Mapping[str, int],
    *,
    minimum_token_frames: int,
    include_prompt: bool,
    checkpoint_id: str,
    root: Path,
) -> list[dict[str, Any]]:
    allowed = callback._classifier_vocabulary_filter()
    cohort = callback._ensure_cohort(
        records,
        artist_to_index,
        minimum_token_frames=minimum_token_frames,
        allowed_artist_keys=allowed,
    )
    artist_offsets: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for sample_index, item in enumerate(cohort):
        offset = artist_offsets[item.artist_key]
        artist_offsets[item.artist_key] += 1
        record = records[item.dataset_index]
        identifiers = _record_identifiers(record, dataset_index=item.dataset_index)
        track_id = str(identifiers["track_id"])
        clip_id = str(identifiers["clip_id"])
        track_slug = _short_slug(track_id, fallback="unknown", limit=32)
        start_token = (
            f"{int(identifiers['start_ms']):09d}"
            if identifiers["start_ms"] is not None
            else "unknown"
        )
        end_token = (
            f"{int(identifiers['end_ms']):09d}"
            if identifiers["end_ms"] is not None
            else "unknown"
        )
        artist_token = _short_slug(
            item.artist_key, fallback=f"artist_{item.artist_index}", limit=24
        )
        sample_id = (
            f"jmdmc__a_{artist_token}__t_{track_slug}__s_{start_token}__"
            f"e_{end_token}__c_{identifiers['caption_sha256_16']}"
        )
        artist_dir = (
            f"artist_{item.artist_index:04d}__"
            f"{_short_slug(item.artist_key, fallback='unknown', limit=32)}"
        )
        clip_dir = root / "audio" / artist_dir / f"sample_{sample_index:04d}"
        file_prefix = (
            f"ckpt-{_short_slug(checkpoint_id, fallback='checkpoint', limit=24)}__"
            f"track-{_short_slug(track_id, fallback='unknown', limit=24)}__"
            f"s-{start_token}__e-{end_token}__"
            f"c-{identifiers['caption_sha256_16']}"
        )
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "artist_clip_index": offset,
            "dataset_index": item.dataset_index,
            "checkpoint_id": checkpoint_id,
            "artist_key": item.artist_key,
            "artist_index": item.artist_index,
            "track_id": track_id,
            "clip_id": clip_id,
            "start_ms": identifiers["start_ms"],
            "end_ms": identifiers["end_ms"],
            "caption_sha256_16": identifiers["caption_sha256_16"],
            "source_audio": str(record.get("audio", "")),
            "source_artist_id": str(record.get("artist_id", "")),
            "source_artist_name": str(record.get("artist_name", "")),
            "source_title": str(record.get("title", "")),
            "text": identifiers["caption"],
            "prompt_path": (
                _relative(clip_dir / f"{file_prefix}__prompt.wav", root)
                if include_prompt
                else ""
            ),
            "reference_path": _relative(
                clip_dir / f"{file_prefix}__reference.wav", root
            ),
        }
        for scenario in SCENARIO_NAMES:
            row[f"{scenario}_path"] = _relative(
                clip_dir / f"{file_prefix}__{scenario}.wav", root
            )
        rows.append(row)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted(
            value for value in set(sample_ids) if sample_ids.count(value) > 1
        )
        raise ValueError(
            "stable sample_id collision; inspect duplicate source/caption rows: "
            + ", ".join(duplicates[:5])
        )
    return rows


def _generation_identity(
    checkpoint_path: Path,
    resolved_config_path: Path,
    callback: EpochControlEvaluationCallback,
    sample_rate: int,
    audio_subtype: str,
    checkpoint_id: str,
    checkpoint_sha256: str,
    config_sha256: str,
    cohort_sha256: str,
    scenario_plan_sha256: str,
    respect_classifier_vocabulary: bool,
) -> dict[str, Any]:
    stat = checkpoint_path.stat()
    stable_identity = {
        "format_version": FORMAT_VERSION,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "cohort_sha256": cohort_sha256,
        "scenario_plan_sha256": scenario_plan_sha256,
        "respect_classifier_vocabulary": bool(respect_classifier_vocabulary),
        "num_artists": callback.num_artists,
        "clips_per_artist": callback.clips_per_artist,
        "generation_batch_size": callback.generation_batch_size,
        "generation_seed": callback.generation_seed,
        "cohort_seed": callback.cohort_seed,
        "generation_mode": callback.generation_mode,
        "audio_prompt_seconds": (
            callback.audio_prompt_seconds
            if callback.generation_mode == "continuation"
            else 0.0
        ),
        "configured_audio_prompt_seconds": callback.audio_prompt_seconds,
        "continuation_seconds": callback.continuation_seconds,
        "multi_min_artists": callback.multi_min_artists,
        "multi_max_artists": callback.multi_max_artists,
        "sample_rate": sample_rate,
        "audio_subtype": audio_subtype,
        "generation_kwargs": callback.generation_kwargs,
        "scenarios": list(SCENARIO_NAMES),
    }
    return {
        **stable_identity,
        "generation_id": _canonical_sha256(stable_identity),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "config_path": str(resolved_config_path),
    }


def _validate_existing_identity(path: Path, identity: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("generation_id") != identity.get("generation_id"):
        raise ValueError(
            "output directory belongs to a different checkpoint/config/cohort/"
            "scenario generation; choose another output_dir or pass overwrite=True"
        )


@torch.inference_mode()
def generate_checkpoint_audio(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    musicgen_model_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    local_files_only: bool = True,
    num_artists: int | None = None,
    clips_per_artist: int | None = None,
    generation_batch_size: int | None = None,
    generation_seed: int | None = None,
    cohort_seed: int | None = None,
    generation_mode: str | None = None,
    audio_prompt_seconds: float | None = None,
    continuation_seconds: float | None = None,
    export_id: str | None = None,
    checkpoint_sha256: str | None = None,
    respect_classifier_vocabulary: bool = True,
    audio_subtype: str = "FLOAT",
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """Restore a checkpoint and stream all six scenario generations to disk.

    Existing complete WAV files are reused unless ``overwrite`` is true.  The
    function resets the RNG to the same seed before each scenario, matching the
    epoch callback so differences are attributable to control rather than a
    different sampling stream.
    """

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint_sha256 = checkpoint_sha256 or _file_sha256(checkpoint)
    checkpoint_id = _short_slug(
        export_id
        or (
            f"{checkpoint.parent.parent.name if len(checkpoint.parents) > 1 else checkpoint.parent.name}__"
            f"{checkpoint.stem}__{resolved_checkpoint_sha256[:12]}"
        ),
        fallback="checkpoint",
        limit=64,
    )
    config, resolved_config_path, datamodule, model = restore_control_runtime(
        checkpoint,
        config_path=config_path,
        musicgen_model_path=musicgen_model_path,
        local_files_only=local_files_only,
        device=device,
    )
    callback = EpochControlEvaluationCallback(
        **_callback_kwargs(
            config,
            num_artists=num_artists,
            clips_per_artist=clips_per_artist,
            generation_batch_size=generation_batch_size,
            generation_seed=generation_seed,
            cohort_seed=cohort_seed,
            generation_mode=generation_mode,
            audio_prompt_seconds=audio_prompt_seconds,
            continuation_seconds=continuation_seconds,
            respect_classifier_vocabulary=respect_classifier_vocabulary,
        )
    )
    try:
        dataset, records, artist_to_index, collator = callback._training_data(
            type("OfflineTrainer", (), {"datamodule": datamodule})()
        )
        model_device = torch.device(getattr(model, "device", device))
        frame_rate = float(audio_frame_rate(model))
        use_audio_prompt = callback.generation_mode == "continuation"
        prompt_frames = (
            max(1, int(round(callback.audio_prompt_seconds * frame_rate)))
            if use_audio_prompt
            else 0
        )
        continuation_frames = max(
            1, int(round(callback.continuation_seconds * frame_rate))
        )
        encoded_prompt_seconds = prompt_frames / frame_rate
        rows = _clip_rows(
            callback,
            dataset,
            records,
            artist_to_index,
            minimum_token_frames=prompt_frames + continuation_frames,
            include_prompt=use_audio_prompt,
            checkpoint_id=checkpoint_id,
            root=root,
        )
        assert callback._cohort is not None and callback._scenario_plan is not None
        cohort = callback._cohort
        sample_rate = int(getattr(model.model, "audio_sample_rate", 32_000))
        num_codebooks = int(getattr(model.model, "num_codebooks", 1) or 1)
        generation_kwargs = dict(callback.generation_kwargs)
        generation_kwargs.setdefault(
            "max_new_tokens",
            int(math.ceil(callback.continuation_seconds * frame_rate)) + num_codebooks,
        )
        callback.generation_kwargs = generation_kwargs
        cohort_sha256 = _canonical_sha256(
            [
                {
                    key: row.get(key)
                    for key in (
                        "sample_id",
                        "artist_key",
                        "track_id",
                        "start_ms",
                        "end_ms",
                        "caption_sha256_16",
                    )
                }
                for row in rows
            ]
        )
        scenario_plan_sha256 = _canonical_sha256(
            [
                {
                    "name": scenario.name,
                    "direction": scenario.direction,
                    "artist_sets": [list(values) for values in scenario.artist_sets],
                }
                for scenario in callback._scenario_plan
            ]
        )
        identity = _generation_identity(
            checkpoint,
            resolved_config_path,
            callback,
            sample_rate,
            audio_subtype,
            checkpoint_id,
            resolved_checkpoint_sha256,
            _file_sha256(resolved_config_path),
            cohort_sha256,
            scenario_plan_sha256,
            respect_classifier_vocabulary,
        )
        metadata_path = root / RUN_METADATA_NAME
        if not overwrite:
            _validate_existing_identity(metadata_path, identity)
        metadata = {
            **identity,
            "status": "generating",
            "num_clips": len(rows),
            "expected_kl_points_per_scenario": len(rows),
            "expected_fad_points_per_scenario": callback.num_artists,
        }
        _json_dump(metadata_path, metadata)
        _write_csv(root / CLIP_MANIFEST_NAME, rows)

        decode = getattr(model.model, "decode_audio_tokens", None)
        generate = getattr(model, "_generate_with_condition", None)
        if not callable(decode) or not callable(generate):
            raise RuntimeError(
                "model must expose model.decode_audio_tokens() and "
                "_generate_with_condition()"
            )
        index_to_artist = {
            int(index): str(key) for key, index in artist_to_index.items()
        }
        condition_rows: list[dict[str, Any]] = []
        generated_manifest: list[dict[str, Any]] = []
        for sample_index in range(len(rows)):
            for scenario in callback._scenario_plan:
                artist_ids = list(scenario.artist_sets[sample_index])
                artist_keys = [index_to_artist[index] for index in artist_ids]
                condition_rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": rows[sample_index]["sample_id"],
                        "track_id": rows[sample_index]["track_id"],
                        "scenario": scenario.name,
                        "direction": scenario.direction,
                        "condition_artist_indices": json.dumps(artist_ids),
                        "condition_artist_keys": json.dumps(
                            artist_keys, ensure_ascii=False
                        ),
                    }
                )
                row = rows[sample_index]
                generated_manifest.append(
                    {
                        "format_version": FORMAT_VERSION,
                        "checkpoint_id": checkpoint_id,
                        "sample_id": row["sample_id"],
                        "sample_index": sample_index,
                        "dataset_index": int(row["dataset_index"]),
                        "artist_key": row["artist_key"],
                        "artist_index": int(row["artist_index"]),
                        "track_id": row["track_id"],
                        "clip_id": row["clip_id"],
                        "start_ms": row["start_ms"],
                        "end_ms": row["end_ms"],
                        "caption_sha256_16": row["caption_sha256_16"],
                        "source_audio": row["source_audio"],
                        "source_artist_id": row["source_artist_id"],
                        "source_artist_name": row["source_artist_name"],
                        "source_title": row["source_title"],
                        "caption": row["text"],
                        "scenario": scenario.name,
                        "scenario_label": SCENARIO_LABELS[scenario.name],
                        "control_direction": scenario.direction,
                        "condition_artist_indices": artist_ids,
                        "condition_artist_keys": artist_keys,
                        "audio_path": row[f"{scenario.name}_path"],
                        "reference_path": row["reference_path"],
                        "prompt_path": row["prompt_path"] or None,
                        "sample_rate": sample_rate,
                        "duration_seconds": callback.continuation_seconds,
                    }
                )
        _write_csv(root / "scenario_conditions.csv", condition_rows)
        _write_jsonl(root / GENERATED_MANIFEST_NAME, generated_manifest)

        from tqdm.auto import tqdm

        total_outputs = len(cohort) * len(SCENARIO_NAMES)
        generated_count = 0
        cached_count = 0
        with tqdm(
            total=total_outputs,
            desc="Generating six-scenario audio",
            unit="audio",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for start in range(0, len(cohort), callback.generation_batch_size):
                stop = min(start + callback.generation_batch_size, len(cohort))
                batch_rows = rows[start:stop]
                required_paths = [
                    _absolute(root, row[f"{scenario}_path"])
                    for row in batch_rows
                    for scenario in SCENARIO_NAMES
                ]
                required_paths.extend(
                    _absolute(root, row["reference_path"]) for row in batch_rows
                )
                if use_audio_prompt:
                    required_paths.extend(
                        _absolute(root, row["prompt_path"]) for row in batch_rows
                    )
                if not overwrite and all(path.is_file() for path in required_paths):
                    reused = len(batch_rows) * len(SCENARIO_NAMES)
                    cached_count += reused
                    progress.update(reused)
                    progress.set_postfix(
                        scenario="cached batch",
                        generated=generated_count,
                        cached=cached_count,
                    )
                    continue

                items = cohort[start:stop]
                cpu_batch = collator([dataset[item.dataset_index] for item in items])
                batch = _move_to_device(cpu_batch, model_device)
                tokens = batch.get("audio_tokens")
                decoder_mask = batch.get("decoder_attention_mask")
                if not isinstance(tokens, Tensor) or not isinstance(
                    decoder_mask, Tensor
                ):
                    raise ValueError(
                        "collator must return audio_tokens and decoder_attention_mask"
                    )
                prompt_missing = use_audio_prompt and any(
                    not _absolute(root, row["prompt_path"]).is_file()
                    for row in batch_rows
                )
                reference_missing = any(
                    not _absolute(root, row["reference_path"]).is_file()
                    for row in batch_rows
                )
                if overwrite or prompt_missing or reference_missing:
                    decoded, _ = decode(tokens, decoder_mask)
                    prompts = (
                        fixed_audio_segment(
                            decoded,
                            sample_rate=sample_rate,
                            start_seconds=0.0,
                            duration_seconds=encoded_prompt_seconds,
                        )
                        if use_audio_prompt
                        else None
                    )
                    references = fixed_audio_segment(
                        decoded,
                        sample_rate=sample_rate,
                        start_seconds=encoded_prompt_seconds,
                        duration_seconds=callback.continuation_seconds,
                    )
                    for offset, row in enumerate(batch_rows):
                        prompt_path = _absolute(root, row["prompt_path"])
                        reference_path = _absolute(root, row["reference_path"])
                        if use_audio_prompt and (
                            overwrite or not prompt_path.is_file()
                        ):
                            assert prompts is not None
                            _write_audio(
                                prompt_path,
                                prompts[offset],
                                sample_rate,
                                audio_subtype,
                            )
                        if overwrite or not reference_path.is_file():
                            _write_audio(
                                reference_path,
                                references[offset],
                                sample_rate,
                                audio_subtype,
                            )
                generation_batch = (
                    attach_audio_prompt(batch, prompt_frames=prompt_frames)
                    if use_audio_prompt
                    else batch
                )
                batch_seed = callback.generation_seed + start
                cuda_devices = []
                if model_device.type == "cuda":
                    cuda_devices = [
                        (
                            model_device.index
                            if model_device.index is not None
                            else torch.cuda.current_device()
                        )
                    ]
                for full_scenario in callback._scenario_plan:
                    progress.set_postfix(
                        scenario=full_scenario.name,
                        generated=generated_count,
                        cached=cached_count,
                    )
                    scenario_paths = [
                        _absolute(root, row[f"{full_scenario.name}_path"])
                        for row in batch_rows
                    ]
                    if not overwrite and all(path.is_file() for path in scenario_paths):
                        cached_count += len(batch_rows)
                        progress.update(len(batch_rows))
                        continue
                    scenario = callback._slice_scenario(
                        full_scenario, start, stop, model_device
                    )
                    condition = callback._condition(model, scenario)
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(batch_seed)
                        generated = generate(
                            generation_batch, generation_kwargs, condition
                        )
                    continuations = fixed_audio_segment(
                        generated,
                        sample_rate=sample_rate,
                        start_seconds=encoded_prompt_seconds,
                        duration_seconds=callback.continuation_seconds,
                    )
                    for offset, path in enumerate(scenario_paths):
                        if overwrite or not path.is_file():
                            _write_audio(
                                path,
                                continuations[offset],
                                sample_rate,
                                audio_subtype,
                            )
                    generated_count += len(batch_rows)
                    progress.update(len(batch_rows))
                    del generated, continuations
            progress.set_postfix(
                scenario="complete",
                generated=generated_count,
                cached=cached_count,
            )

        metadata["status"] = "complete"
        metadata["audio_assets_fingerprint"] = _asset_fingerprint(root, rows)
        _json_dump(metadata_path, metadata)
        return root
    finally:
        del model, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _release_runtime(runtime: Any) -> None:
    move = getattr(runtime, "to", None)
    if callable(move):
        move("cpu")
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_kl_distribution(
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    batch_size: int = 8,
    runtime: Any | None = None,
    overwrite: bool = False,
) -> Path:
    """Compute one paired PaSST KL value per clip and per scenario."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(output_dir).expanduser().resolve()
    target = root / "metrics" / KL_VALUES_NAME
    metadata, rows = _complete_generation(root)
    runtime_id = (
        "project.PaSSTKLDivergence:v1"
        if runtime is None
        else f"injected:{type(runtime).__module__}.{type(runtime).__qualname__}"
    )
    if _metric_cache(
        target,
        generation_id=str(metadata["generation_id"]),
        audio_assets_fingerprint=str(metadata["audio_assets_fingerprint"]),
        runtime_id=runtime_id,
        overwrite=overwrite,
    ):
        return target
    sample_rate = int(metadata["sample_rate"])
    expected_frames = int(round(sample_rate * float(metadata["continuation_seconds"])))
    if runtime is None:
        from src.metric.quality.passt_kl_divergence import PaSSTKLDivergence

        runtime = PaSSTKLDivergence()
    if isinstance(runtime, torch.nn.Module):
        runtime.to(device).eval()
    values: list[dict[str, Any]] = []
    try:
        for scenario in SCENARIO_NAMES:
            for start in range(0, len(rows), batch_size):
                selected = rows[start : start + batch_size]
                reference = _stack_audio(
                    [_absolute(root, row["reference_path"]) for row in selected],
                    sample_rate=sample_rate,
                    expected_frames=expected_frames,
                )
                generated = _stack_audio(
                    [_absolute(root, row[f"{scenario}_path"]) for row in selected],
                    sample_rate=sample_rate,
                    expected_frames=expected_frames,
                )
                scores = _passt_per_sample(
                    runtime, generated, reference, sample_rate=sample_rate
                )
                if scores.numel() != len(selected):
                    raise ValueError("PaSST must return one KL value per clip")
                for row, score in zip(selected, scores.tolist()):
                    values.append(
                        {
                            "scenario": scenario,
                            "scenario_label": SCENARIO_LABELS[scenario],
                            **_clip_identity_fields(row),
                            "kl_reference_to_generated": float(score),
                        }
                    )
        _write_csv(target, values)
        _write_metric_sidecar(
            target,
            generation_id=str(metadata["generation_id"]),
            audio_assets_fingerprint=str(metadata["audio_assets_fingerprint"]),
            runtime_id=runtime_id,
            row_count=len(values),
        )
        return target
    finally:
        _release_runtime(runtime)


def compute_fad_distribution(
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    clip_batch_size: int = 8,
    runtime: Any | None = None,
    overwrite: bool = False,
) -> Path:
    """Compute one group-to-group VGGish FAD per artist and scenario."""

    if clip_batch_size <= 0:
        raise ValueError("clip_batch_size must be positive")
    root = Path(output_dir).expanduser().resolve()
    target = root / "metrics" / FAD_VALUES_NAME
    metadata, rows = _complete_generation(root)
    runtime_id = (
        "project.VGGishAudioEmbedding:preactivation-v1"
        if runtime is None
        else f"injected:{type(runtime).__module__}.{type(runtime).__qualname__}"
    )
    if _metric_cache(
        target,
        generation_id=str(metadata["generation_id"]),
        audio_assets_fingerprint=str(metadata["audio_assets_fingerprint"]),
        runtime_id=runtime_id,
        overwrite=overwrite,
    ):
        return target
    sample_rate = int(metadata["sample_rate"])
    expected_frames = int(round(sample_rate * float(metadata["continuation_seconds"])))
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["artist_index"]), row["artist_key"])].append(row)
    if runtime is None:
        from src.metric.quality.vggish_embedding import VGGishAudioEmbedding

        runtime = VGGishAudioEmbedding()
    if isinstance(runtime, torch.nn.Module):
        runtime.to(device).eval()

    def embeddings(paths: Sequence[Path]) -> Tensor:
        chunks: list[Tensor] = []
        for start in range(0, len(paths), clip_batch_size):
            audio = _stack_audio(
                paths[start : start + clip_batch_size],
                sample_rate=sample_rate,
                expected_frames=expected_frames,
            )
            chunks.append(_vggish_embeddings(runtime, audio, sample_rate=sample_rate))
        return torch.cat(chunks, dim=0)

    values: list[dict[str, Any]] = []
    try:
        for (artist_index, artist_key), artist_rows in sorted(grouped.items()):
            reference_embeddings = embeddings(
                [_absolute(root, row["reference_path"]) for row in artist_rows]
            )
            for scenario in SCENARIO_NAMES:
                generated_embeddings = embeddings(
                    [_absolute(root, row[f"{scenario}_path"]) for row in artist_rows]
                )
                fad = frechet_distance_from_embeddings(
                    generated_embeddings, reference_embeddings
                )
                values.append(
                    {
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "artist_key": artist_key,
                        "artist_index": artist_index,
                        "checkpoint_id": artist_rows[0].get("checkpoint_id", ""),
                        "num_clips": len(artist_rows),
                        "num_reference_windows": reference_embeddings.shape[0],
                        "num_generated_windows": generated_embeddings.shape[0],
                        "fad": float(fad.item()),
                    }
                )
        _write_csv(target, values)
        _write_metric_sidecar(
            target,
            generation_id=str(metadata["generation_id"]),
            audio_assets_fingerprint=str(metadata["audio_assets_fingerprint"]),
            runtime_id=runtime_id,
            row_count=len(values),
        )
        return target
    finally:
        _release_runtime(runtime)


def compute_artist_attribution_distribution(
    output_dir: str | Path,
    *,
    classifier_checkpoint_path: str | Path | None = None,
    classifier_config_path: str | Path | None = None,
    classifier_model_name_or_path: str | Path | None = None,
    local_files_only: bool = True,
    device: str | torch.device = "cuda",
    batch_size: int = 8,
    runtime: Any | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist per-clip target-artist hit, confidence, rank and prediction."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(output_dir).expanduser().resolve()
    target = root / "metrics" / ATTRIBUTION_VALUES_NAME
    artist_target = root / "metrics" / ATTRIBUTION_ARTIST_VALUES_NAME
    metadata, rows = _complete_generation(root)
    runtime_id = (
        (
            "project.ArtistClassifierRuntime:"
            + _file_sha256(Path(classifier_checkpoint_path).expanduser().resolve())
        )
        if runtime is None and classifier_checkpoint_path is not None
        else (
            f"injected:{type(runtime).__module__}.{type(runtime).__qualname__}"
            if runtime is not None
            else "project.ArtistClassifierRuntime:missing"
        )
    )
    clip_cached = _metric_cache(
        target,
        generation_id=str(metadata["generation_id"]),
        audio_assets_fingerprint=str(metadata["audio_assets_fingerprint"]),
        runtime_id=runtime_id,
        overwrite=overwrite,
    )
    artist_cached = _metric_cache(
        artist_target,
        generation_id=str(metadata["generation_id"]),
        audio_assets_fingerprint=str(metadata["audio_assets_fingerprint"]),
        runtime_id=runtime_id,
        overwrite=overwrite,
    )
    if clip_cached and artist_cached:
        return target
    sample_rate = int(metadata["sample_rate"])
    expected_frames = int(round(sample_rate * float(metadata["continuation_seconds"])))
    if runtime is None:
        if classifier_checkpoint_path is None:
            raise ValueError(
                "classifier_checkpoint_path is required unless runtime is injected"
            )
        from src.evaluation.metric_runtimes import ArtistClassifierRuntime

        runtime = ArtistClassifierRuntime.from_checkpoint(
            classifier_checkpoint_path,
            config_path=classifier_config_path,
            model_name_or_path=classifier_model_name_or_path,
            local_files_only=local_files_only,
            device=device,
        )
    move = getattr(runtime, "to", None)
    if callable(move):
        move(device)
    classifier_indices = getattr(runtime, "classifier_indices", None)
    logits_method = getattr(runtime, "logits", None)
    if not callable(classifier_indices) or not callable(logits_method):
        raise TypeError(
            "classifier runtime must provide classifier_indices() and logits()"
        )
    target_indices = torch.as_tensor(
        classifier_indices([row["artist_key"] for row in rows])
    ).long()
    vocabulary = tuple(
        str(value) for value in getattr(runtime, "artist_vocabulary", ())
    )
    values: list[dict[str, Any]] = []
    try:
        from src.evaluation.metric_runtimes import classifier_target_statistics

        for scenario in SCENARIO_NAMES:
            for start in range(0, len(rows), batch_size):
                selected = rows[start : start + batch_size]
                generated = _stack_audio(
                    [_absolute(root, row[f"{scenario}_path"]) for row in selected],
                    sample_rate=sample_rate,
                    expected_frames=expected_frames,
                )
                logits = torch.as_tensor(
                    logits_method(
                        generated,
                        sample_rate=sample_rate,
                        batch_size=batch_size,
                    )
                ).float()
                targets = target_indices[start : start + len(selected)]
                stats = classifier_target_statistics(logits, targets)
                predicted = logits.argmax(dim=-1).cpu()
                for offset, row in enumerate(selected):
                    predicted_index = int(predicted[offset].item())
                    predicted_key = (
                        vocabulary[predicted_index]
                        if 0 <= predicted_index < len(vocabulary)
                        else str(predicted_index)
                    )
                    values.append(
                        {
                            "scenario": scenario,
                            "scenario_label": SCENARIO_LABELS[scenario],
                            **_clip_identity_fields(row),
                            # This is a per-clip 0/1 hit. Its cohort mean is the rate.
                            "target_attribution_hit": float(
                                stats.attribution[offset].item()
                            ),
                            "target_artist_confidence": float(
                                stats.confidence[offset].item()
                            ),
                            "target_artist_rank": float(stats.rank[offset].item()),
                            "predicted_artist_index": predicted_index,
                            "predicted_artist_key": predicted_key,
                        }
                    )
        _write_csv(target, values)
        _write_metric_sidecar(
            target,
            generation_id=str(metadata["generation_id"]),
            audio_assets_fingerprint=str(metadata["audio_assets_fingerprint"]),
            runtime_id=runtime_id,
            row_count=len(values),
        )
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            grouped[(str(row["scenario"]), str(row["artist_key"]))].append(row)
        artist_values: list[dict[str, Any]] = []
        for (scenario, artist_key), group in sorted(grouped.items()):
            artist_values.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "checkpoint_id": group[0]["checkpoint_id"],
                    "artist_key": artist_key,
                    "artist_index": int(group[0]["artist_index"]),
                    "num_clips": len(group),
                    "target_attribution_rate": sum(
                        float(row["target_attribution_hit"]) for row in group
                    )
                    / len(group),
                    "mean_target_artist_confidence": sum(
                        float(row["target_artist_confidence"]) for row in group
                    )
                    / len(group),
                    "mean_target_artist_rank": sum(
                        float(row["target_artist_rank"]) for row in group
                    )
                    / len(group),
                }
            )
        _write_csv(artist_target, artist_values)
        _write_metric_sidecar(
            artist_target,
            generation_id=str(metadata["generation_id"]),
            audio_assets_fingerprint=str(metadata["audio_assets_fingerprint"]),
            runtime_id=runtime_id,
            row_count=len(artist_values),
        )
        return target
    finally:
        _release_runtime(runtime)


def compute_metric_distributions(
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    batch_size: int = 8,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Compute KL first, release PaSST, then compute FAD with VGGish."""

    kl_path = compute_kl_distribution(
        output_dir, device=device, batch_size=batch_size, overwrite=overwrite
    )
    fad_path = compute_fad_distribution(
        output_dir, device=device, clip_batch_size=batch_size, overwrite=overwrite
    )
    return kl_path, fad_path


def _ranking_scores(
    rows: Sequence[Mapping[str, str]],
    *,
    group_key: str,
    value_key: str,
    ranking_scenario: str,
) -> list[tuple[str, float]]:
    if (
        ranking_scenario != "mean_all_scenarios"
        and ranking_scenario not in SCENARIO_NAMES
    ):
        raise ValueError(
            "ranking_scenario must be mean_all_scenarios or one of " f"{SCENARIO_NAMES}"
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            ranking_scenario != "mean_all_scenarios"
            and row["scenario"] != ranking_scenario
        ):
            continue
        grouped[str(row[group_key])].append(float(row[value_key]))
    return sorted(
        ((key, sum(values) / len(values)) for key, values in grouped.items()),
        key=lambda item: item[1],
    )


def _playlist(path: Path, audio_paths: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#EXTM3U\n" + "\n".join(str(item.resolve()) for item in audio_paths) + "\n",
        encoding="utf-8",
    )


def build_best_worst_rankings(
    output_dir: str | Path,
    *,
    top_k: int = 10,
    ranking_scenario: str = "suppress_single_target",
) -> Path:
    """Save best/worst group manifests and six-scenario playlists.

    Lower KL/FAD is considered better.  KL groups are clips; FAD groups are
    artists and contain a playlist for every clip.  Audio remains in the
    canonical ``audio/artist/clip`` tree, so ranking indexes do not duplicate
    gigabytes of WAV data.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    root = Path(output_dir).expanduser().resolve()
    clips = _read_csv(root / CLIP_MANIFEST_NAME)
    clips_by_index = {row["sample_index"]: row for row in clips}
    clips_by_artist: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clips:
        clips_by_artist[row["artist_key"]].append(row)
    metric_specs = {
        "kl": (
            _read_csv(root / "metrics" / KL_VALUES_NAME),
            "sample_index",
            "kl_reference_to_generated",
        ),
        "fad": (_read_csv(root / "metrics" / FAD_VALUES_NAME), "artist_key", "fad"),
    }
    ranking_root = root / "rankings" / ranking_scenario
    index: dict[str, Any] = {
        "ranking_scenario": ranking_scenario,
        "top_k": top_k,
        "lower_is_better": True,
        "metrics": {},
    }
    for metric, (metric_rows, key_name, value_name) in metric_specs.items():
        scores = _ranking_scores(
            metric_rows,
            group_key=key_name,
            value_key=value_name,
            ranking_scenario=ranking_scenario,
        )
        index["metrics"][metric] = {}
        selections = {"best": scores[:top_k], "worst": list(reversed(scores[-top_k:]))}
        for tail, selected in selections.items():
            saved: list[dict[str, Any]] = []
            for rank, (key, score) in enumerate(selected, start=1):
                group_dir = ranking_root / metric / tail / f"rank_{rank:02d}"
                if metric == "kl":
                    group_clips = [clips_by_index[key]]
                else:
                    group_clips = sorted(
                        clips_by_artist[key],
                        key=lambda row: int(row["artist_clip_index"]),
                    )
                clip_entries: list[dict[str, Any]] = []
                for clip in group_clips:
                    scenario_paths = {
                        scenario: str(_absolute(root, clip[f"{scenario}_path"]))
                        for scenario in SCENARIO_NAMES
                    }
                    playlist_path = group_dir / (
                        f"clip_{int(clip['artist_clip_index']):04d}.m3u8"
                    )
                    _playlist(
                        playlist_path,
                        (Path(scenario_paths[name]) for name in SCENARIO_NAMES),
                    )
                    clip_entries.append(
                        {
                            "sample_index": int(clip["sample_index"]),
                            "artist_clip_index": int(clip["artist_clip_index"]),
                            "text": clip["text"],
                            "reference_path": str(
                                _absolute(root, clip["reference_path"])
                            ),
                            "prompt_path": (
                                str(_absolute(root, clip["prompt_path"]))
                                if clip.get("prompt_path")
                                else None
                            ),
                            "scenario_paths": scenario_paths,
                            "playlist_path": str(playlist_path.resolve()),
                        }
                    )
                entry = {
                    "rank": rank,
                    "group_key": key,
                    "score": score,
                    "artist_key": group_clips[0]["artist_key"],
                    "clips": clip_entries,
                }
                _json_dump(group_dir / "group.json", entry)
                saved.append(entry)
            index["metrics"][metric][tail] = saved
    _json_dump(ranking_root / "index.json", index)
    return ranking_root / "index.json"


def create_dummy_evaluation(
    output_dir: str | Path,
    *,
    num_artists: int = 3,
    clips_per_artist: int = 4,
    sample_rate: int = 8_000,
    duration_seconds: float = 0.25,
) -> Path:
    """Create a tiny deterministic artifact tree for notebook/tests only."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(7)
    rows: list[dict[str, Any]] = []
    frames = int(round(sample_rate * duration_seconds))
    time = torch.arange(frames) / sample_rate
    sample_index = 0
    checkpoint_id = "dummy-checkpoint"
    generated_manifest: list[dict[str, Any]] = []
    for artist_index in range(num_artists):
        artist_key = f"dummy_artist_{artist_index:02d}"
        for clip_index in range(clips_per_artist):
            track_id = f"dummy_track_{artist_index:02d}_{clip_index:03d}"
            caption = f"dummy caption {sample_index}"
            caption_hash = hashlib.sha256(caption.encode("utf-8")).hexdigest()[:16]
            sample_id = (
                f"jmdmc__a_{artist_key}__t_{track_id}__s_000000000__"
                f"e_000000250__c_{caption_hash}"
            )
            clip_dir = (
                root
                / "audio"
                / f"artist_{artist_index:04d}__{artist_key}"
                / f"sample_{sample_index:04d}"
            )
            prefix = (
                f"ckpt-{checkpoint_id}__track-{track_id}__s-000000000__"
                f"e-000000250__c-{caption_hash}"
            )
            frequency = 160.0 + artist_index * 40.0 + clip_index * 5.0
            reference = 0.15 * torch.sin(2 * math.pi * frequency * time)
            prompt = reference[: frames // 2]
            row: dict[str, Any] = {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "artist_clip_index": clip_index,
                "dataset_index": sample_index,
                "checkpoint_id": checkpoint_id,
                "artist_key": artist_key,
                "artist_index": artist_index,
                "track_id": track_id,
                "clip_id": f"dummy_clip_{sample_index:04d}",
                "start_ms": 0,
                "end_ms": 250,
                "caption_sha256_16": caption_hash,
                "source_audio": f"dummy/{track_id}.wav",
                "source_artist_id": str(artist_index),
                "source_artist_name": artist_key,
                "source_title": f"Dummy track {clip_index}",
                "text": caption,
                "prompt_path": _relative(clip_dir / f"{prefix}__prompt.wav", root),
                "reference_path": _relative(
                    clip_dir / f"{prefix}__reference.wav", root
                ),
            }
            _write_audio(
                _absolute(root, row["prompt_path"]), prompt, sample_rate, "FLOAT"
            )
            _write_audio(
                _absolute(root, row["reference_path"]),
                reference,
                sample_rate,
                "FLOAT",
            )
            for scenario_index, scenario in enumerate(SCENARIO_NAMES):
                path = clip_dir / f"{prefix}__{scenario}.wav"
                noise = (
                    torch.randn(frames, generator=generator)
                    * (scenario_index + 1)
                    * 0.002
                )
                _write_audio(path, reference + noise, sample_rate, "FLOAT")
                row[f"{scenario}_path"] = _relative(path, root)
                generated_manifest.append(
                    {
                        "format_version": FORMAT_VERSION,
                        "checkpoint_id": checkpoint_id,
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "artist_key": artist_key,
                        "artist_index": artist_index,
                        "track_id": track_id,
                        "caption": caption,
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "audio_path": row[f"{scenario}_path"],
                        "reference_path": row["reference_path"],
                        "prompt_path": row["prompt_path"],
                        "sample_rate": sample_rate,
                        "duration_seconds": duration_seconds,
                    }
                )
            rows.append(row)
            sample_index += 1
    _write_csv(root / CLIP_MANIFEST_NAME, rows)
    _write_jsonl(root / GENERATED_MANIFEST_NAME, generated_manifest)
    stable_identity = {
        "format_version": FORMAT_VERSION,
        "checkpoint_id": checkpoint_id,
        "num_artists": num_artists,
        "clips_per_artist": clips_per_artist,
        "sample_rate": sample_rate,
        "continuation_seconds": duration_seconds,
        "cohort_sha256": _canonical_sha256([row["sample_id"] for row in rows]),
    }
    metadata = {
        **stable_identity,
        "generation_id": _canonical_sha256(stable_identity),
        "status": "complete",
        "checkpoint_path": "DUMMY",
        "num_clips": len(rows),
        "generation_mode": "continuation",
        "audio_prompt_seconds": duration_seconds / 2,
        "configured_audio_prompt_seconds": duration_seconds / 2,
        "scenarios": list(SCENARIO_NAMES),
        "audio_assets_fingerprint": _asset_fingerprint(root, rows),
    }
    _json_dump(root / RUN_METADATA_NAME, metadata)
    return root


__all__ = [
    "ATTRIBUTION_ARTIST_VALUES_NAME",
    "ATTRIBUTION_VALUES_NAME",
    "CLIP_MANIFEST_NAME",
    "FAD_VALUES_NAME",
    "GENERATED_MANIFEST_NAME",
    "KL_VALUES_NAME",
    "RUN_METADATA_NAME",
    "build_best_worst_rankings",
    "compute_artist_attribution_distribution",
    "compute_fad_distribution",
    "compute_kl_distribution",
    "compute_metric_distributions",
    "create_dummy_evaluation",
    "generate_checkpoint_audio",
]
