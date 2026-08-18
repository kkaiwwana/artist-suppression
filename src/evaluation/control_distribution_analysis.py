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
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

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


FORMAT_VERSION = 1
CLIP_MANIFEST_NAME = "clips.csv"
RUN_METADATA_NAME = "run_metadata.json"
KL_VALUES_NAME = "passt_kl_per_clip.csv"
FAD_VALUES_NAME = "fad_per_artist.csv"


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
        raise ValueError(f"one audio item must have shape [T] or [C,T], got {audio.shape}")
    sf.write(path, audio.numpy(), int(sample_rate), format="WAV", subtype=subtype)


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
    accepted = set(inspect.signature(EpochControlEvaluationCallback.__init__).parameters)
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
        artist_dir = (
            f"artist_{item.artist_index:04d}__"
            f"{_slug(item.artist_key, fallback='unknown')}"
        )
        clip_dir = root / "audio" / artist_dir / f"clip_{offset:04d}"
        record = records[item.dataset_index]
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "artist_clip_index": offset,
            "dataset_index": item.dataset_index,
            "artist_key": item.artist_key,
            "artist_index": item.artist_index,
            "text": str(record.get("text", record.get("caption", ""))),
            "prompt_path": (
                _relative(clip_dir / "prompt.wav", root) if include_prompt else ""
            ),
            "reference_path": _relative(clip_dir / "reference.wav", root),
        }
        for scenario in SCENARIO_NAMES:
            row[f"{scenario}_path"] = _relative(clip_dir / f"{scenario}.wav", root)
        rows.append(row)
    return rows


def _generation_identity(
    checkpoint_path: Path,
    resolved_config_path: Path,
    callback: EpochControlEvaluationCallback,
    sample_rate: int,
    audio_subtype: str,
) -> dict[str, Any]:
    stat = checkpoint_path.stat()
    return {
        "format_version": FORMAT_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "config_path": str(resolved_config_path),
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


def _validate_existing_identity(path: Path, identity: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    for key, value in identity.items():
        if previous.get(key) != value:
            raise ValueError(
                f"output directory belongs to a different evaluation ({key}: "
                f"{previous.get(key)!r} != {value!r}); choose another output_dir "
                "or pass overwrite=True"
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
        identity = _generation_identity(
            checkpoint, resolved_config_path, callback, sample_rate, audio_subtype
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
        index_to_artist = {int(index): str(key) for key, index in artist_to_index.items()}
        condition_rows: list[dict[str, Any]] = []
        for sample_index in range(len(rows)):
            for scenario in callback._scenario_plan:
                artist_ids = list(scenario.artist_sets[sample_index])
                condition_rows.append(
                    {
                        "sample_index": sample_index,
                        "scenario": scenario.name,
                        "direction": scenario.direction,
                        "condition_artist_indices": json.dumps(artist_ids),
                        "condition_artist_keys": json.dumps(
                            [index_to_artist[index] for index in artist_ids],
                            ensure_ascii=False,
                        ),
                    }
                )
        _write_csv(root / "scenario_conditions.csv", condition_rows)

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
                    _absolute(root, row["reference_path"])
                    for row in batch_rows
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
                        model_device.index
                        if model_device.index is not None
                        else torch.cuda.current_device()
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
                    if not overwrite and all(
                        path.is_file() for path in scenario_paths
                    ):
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
        _json_dump(metadata_path, metadata)
        return root
    finally:
        del model, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _release_runtime(runtime: Any) -> None:
    if isinstance(runtime, torch.nn.Module):
        runtime.to("cpu")
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
    if target.is_file() and not overwrite:
        return target
    metadata = json.loads((root / RUN_METADATA_NAME).read_text(encoding="utf-8"))
    rows = _read_csv(root / CLIP_MANIFEST_NAME)
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
                            "sample_index": int(row["sample_index"]),
                            "artist_key": row["artist_key"],
                            "artist_index": int(row["artist_index"]),
                            "artist_clip_index": int(row["artist_clip_index"]),
                            "kl_reference_to_generated": float(score),
                        }
                    )
        _write_csv(target, values)
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
    if target.is_file() and not overwrite:
        return target
    metadata = json.loads((root / RUN_METADATA_NAME).read_text(encoding="utf-8"))
    rows = _read_csv(root / CLIP_MANIFEST_NAME)
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
                        "num_clips": len(artist_rows),
                        "num_reference_windows": reference_embeddings.shape[0],
                        "num_generated_windows": generated_embeddings.shape[0],
                        "fad": float(fad.item()),
                    }
                )
        _write_csv(target, values)
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
    if ranking_scenario != "mean_all_scenarios" and ranking_scenario not in SCENARIO_NAMES:
        raise ValueError(
            "ranking_scenario must be mean_all_scenarios or one of "
            f"{SCENARIO_NAMES}"
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if ranking_scenario != "mean_all_scenarios" and row["scenario"] != ranking_scenario:
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
                        clips_by_artist[key], key=lambda row: int(row["artist_clip_index"])
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
                            "reference_path": str(_absolute(root, clip["reference_path"])),
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
    for artist_index in range(num_artists):
        artist_key = f"dummy_artist_{artist_index:02d}"
        for clip_index in range(clips_per_artist):
            clip_dir = root / "audio" / f"artist_{artist_index:04d}__{artist_key}" / f"clip_{clip_index:04d}"
            frequency = 160.0 + artist_index * 40.0 + clip_index * 5.0
            reference = 0.15 * torch.sin(2 * math.pi * frequency * time)
            prompt = reference[: frames // 2]
            row: dict[str, Any] = {
                "sample_index": sample_index,
                "artist_clip_index": clip_index,
                "dataset_index": sample_index,
                "artist_key": artist_key,
                "artist_index": artist_index,
                "text": f"dummy caption {sample_index}",
                "prompt_path": _relative(clip_dir / "prompt.wav", root),
                "reference_path": _relative(clip_dir / "reference.wav", root),
            }
            _write_audio(clip_dir / "prompt.wav", prompt, sample_rate, "FLOAT")
            _write_audio(clip_dir / "reference.wav", reference, sample_rate, "FLOAT")
            for scenario_index, scenario in enumerate(SCENARIO_NAMES):
                path = clip_dir / f"{scenario}.wav"
                noise = torch.randn(frames, generator=generator) * (scenario_index + 1) * 0.002
                _write_audio(path, reference + noise, sample_rate, "FLOAT")
                row[f"{scenario}_path"] = _relative(path, root)
            rows.append(row)
            sample_index += 1
    _write_csv(root / CLIP_MANIFEST_NAME, rows)
    metadata = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "checkpoint_path": "DUMMY",
        "num_artists": num_artists,
        "clips_per_artist": clips_per_artist,
        "num_clips": len(rows),
        "sample_rate": sample_rate,
        "generation_mode": "continuation",
        "audio_prompt_seconds": duration_seconds / 2,
        "configured_audio_prompt_seconds": duration_seconds / 2,
        "continuation_seconds": duration_seconds,
        "scenarios": list(SCENARIO_NAMES),
    }
    _json_dump(root / RUN_METADATA_NAME, metadata)
    return root


__all__ = [
    "CLIP_MANIFEST_NAME",
    "FAD_VALUES_NAME",
    "KL_VALUES_NAME",
    "RUN_METADATA_NAME",
    "build_best_worst_rankings",
    "compute_fad_distribution",
    "compute_kl_distribution",
    "compute_metric_distributions",
    "create_dummy_evaluation",
    "generate_checkpoint_audio",
]
