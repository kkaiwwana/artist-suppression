"""Generate paired audio for training-time artist-suppression evaluation.

For every artist known by an unlearning checkpoint, this script samples a
training caption and produces a paired trio with identical sampling seeds:

* ``default``: no intervention;
* ``suppress_self``: subtract the caption artist's condition;
* ``suppress_other``: subtract a deterministic random different artist.

The output manifest is directly consumable by ArtistClassificationDataModule.
Interrupted runs resume at complete trio boundaries by default.

With ``--continuation``, each trio receives the same prefix from its source
clip as an EnCodec audio prompt. The decoded real prefix is removed before
storage, so downstream metrics see only the newly generated continuation.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.evaluation.artist_audio_generation import (
    SAMPLE_TYPES,
    build_generation_jobs,
    fixed_audio_duration,
    load_completed_group_ids,
    manifest_rows_for_job,
)
from src.evaluation.checkpoint_runtime import restore_control_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--musicgen-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--groups-per-artist", type=int, default=2)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument(
        "--continuation",
        action="store_true",
        help=(
            "Condition generation on the beginning of each source clip. Only "
            "the newly generated continuation is saved."
        ),
    )
    parser.add_argument(
        "--audio-prompt-seconds",
        type=float,
        default=5.0,
        help="Source-audio prefix length used when --continuation is enabled.",
    )
    parser.add_argument(
        "--continuation-seconds",
        type=float,
        default=10.0,
        help="Generated tail length saved when --continuation is enabled.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-artists", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Autocast precision used during autoregressive generation.",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a new manifest and overwrite matching audio paths.",
    )
    return parser.parse_args()


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _seed_generation(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _batch_seed(seed: int, group_ids: list[str]) -> int:
    payload = f"{seed}:" + ",".join(group_ids)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--precision bf16 requested but this GPU does not support it")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _audio_frame_rate(model: Any) -> float:
    config = getattr(getattr(model.model, "backbone", None), "config", None)
    audio_config = getattr(config, "audio_encoder", None)
    sampling_rate = float(getattr(audio_config, "sampling_rate", 32_000))
    ratios = getattr(audio_config, "upsampling_ratios", None)
    if ratios:
        return sampling_rate / math.prod(int(value) for value in ratios)
    return 50.0


def _duration_slug(seconds: float) -> str:
    return f"{float(seconds):g}".replace(".", "p")


def _default_output_dir(
    checkpoint: Path,
    *,
    continuation: bool = False,
    audio_prompt_seconds: float = 5.0,
    continuation_seconds: float = 10.0,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    run_name = checkpoint.parent.parent.name
    root = (
        PROJECT_ROOT
        / "evaluation_outputs"
        / "classifier_audio"
        / run_name
        / checkpoint.stem
    )
    if not continuation:
        return root
    suffix = (
        f"continuation_p{_duration_slug(audio_prompt_seconds)}"
        f"_c{_duration_slug(continuation_seconds)}"
    )
    return root / suffix


def _with_audio_continuation_prompt(
    batch: Mapping[str, Any],
    *,
    prompt_frames: int,
) -> dict[str, Any]:
    """Attach fixed-length EnCodec prompt tokens as generation inputs."""

    if prompt_frames <= 0:
        raise ValueError("prompt_frames must be positive")
    audio_tokens = batch.get("audio_tokens")
    if not isinstance(audio_tokens, torch.Tensor) or audio_tokens.ndim != 3:
        raise ValueError("continuation requires audio_tokens shaped [B,Q,T]")
    decoder_mask = batch.get("decoder_attention_mask")
    if not isinstance(decoder_mask, torch.Tensor) or decoder_mask.ndim != 2:
        raise ValueError(
            "continuation requires decoder_attention_mask shaped [B,T]"
        )
    if audio_tokens.shape[0] != decoder_mask.shape[0]:
        raise ValueError("audio token and decoder-mask batch sizes do not match")
    valid_frames = decoder_mask.long().sum(dim=-1)
    if torch.any(valid_frames < prompt_frames):
        shortest = int(valid_frames.min().item())
        raise ValueError(
            f"audio prompt needs {prompt_frames} token frames but the shortest "
            f"source clip has {shortest}"
        )
    batch_size, codebooks, _ = audio_tokens.shape
    decoder_input_ids = audio_tokens[..., :prompt_frames].reshape(
        batch_size * codebooks,
        prompt_frames,
    )
    generation_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "decoder_input_ids": decoder_input_ids.contiguous(),
    }
    prompted = dict(batch)
    prompted["generation_inputs"] = generation_inputs
    return prompted


def _only_generated_continuation(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    prompt_seconds: float,
    continuation_seconds: float,
) -> torch.Tensor:
    """Remove the decoded real prefix and retain an exact generated tail."""

    if audio.ndim == 2:
        audio = audio.unsqueeze(1)
    if audio.ndim != 3:
        raise ValueError("audio must have shape [B,T] or [B,C,T]")
    prompt_samples = int(round(float(prompt_seconds) * int(sample_rate)))
    if prompt_samples <= 0 or prompt_samples >= audio.shape[-1]:
        raise ValueError(
            "generated audio is not longer than its requested audio prompt"
        )
    return fixed_audio_duration(
        audio[..., prompt_samples:],
        sample_rate,
        continuation_seconds,
    )


def _rewrite_manifest_with_complete_groups(manifest: Path) -> set[str]:
    completed = load_completed_group_ids(manifest)
    if not manifest.is_file():
        return completed
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row["group_id"]) in completed:
                    rows.append(row)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return completed


def main() -> None:
    args = parse_args()
    if args.groups_per_artist <= 0 or args.batch_size <= 0:
        raise ValueError("groups-per-artist and batch-size must be positive")
    if args.duration_seconds <= 0:
        raise ValueError("duration-seconds must be positive")
    if args.audio_prompt_seconds <= 0 or args.continuation_seconds <= 0:
        raise ValueError(
            "audio-prompt-seconds and continuation-seconds must be positive"
        )

    generation_mode = "continuation" if args.continuation else "text"
    output_duration_seconds = (
        args.continuation_seconds if args.continuation else args.duration_seconds
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(
            checkpoint,
            continuation=args.continuation,
            audio_prompt_seconds=args.audio_prompt_seconds,
            continuation_seconds=args.continuation_seconds,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_checkpoint = Path(
            str(existing_metadata.get("checkpoint_path", ""))
        ).expanduser().resolve()
        if existing_checkpoint != checkpoint:
            raise ValueError(
                f"output directory belongs to checkpoint {existing_checkpoint}; "
                "choose another --output-dir or pass --overwrite"
            )
        resume_settings = {
            "groups_per_artist": args.groups_per_artist,
            "duration_seconds": output_duration_seconds,
            "seed": args.seed,
            "max_artists": args.max_artists,
            "generation_mode": generation_mode,
            "audio_prompt_seconds": (
                args.audio_prompt_seconds if args.continuation else 0.0
            ),
        }
        for field, expected in resume_settings.items():
            existing = existing_metadata.get(field)
            if field == "generation_mode" and existing is None:
                existing = "text"
            if field == "audio_prompt_seconds" and existing is None:
                existing = 0.0
            if existing != expected:
                raise ValueError(
                    f"resume setting {field}={expected!r} differs from existing "
                    f"{existing!r}; pass --overwrite"
                )
    if args.overwrite:
        manifest_path.write_text("", encoding="utf-8")
        completed: set[str] = set()
    else:
        completed = _rewrite_manifest_with_complete_groups(manifest_path)

    device = torch.device(args.device)
    config, config_path, datamodule, model = restore_control_runtime(
        checkpoint,
        config_path=args.config,
        musicgen_model_path=args.musicgen_model,
        local_files_only=args.local_files_only,
        device=device,
    )
    dataset = datamodule.train_dataset
    if dataset is None or datamodule.vocabulary is None or datamodule._collator is None:
        raise RuntimeError("the restored DataModule did not build its training dataset")
    records = list(dataset.records)
    vocabulary = datamodule.vocabulary.artist_to_index
    jobs = build_generation_jobs(
        records,
        vocabulary,
        groups_per_artist=args.groups_per_artist,
        seed=args.seed,
        max_artists=args.max_artists,
    )
    jobs = [job for job in jobs if job.group_id not in completed]

    frame_rate = _audio_frame_rate(model)
    prompt_frames = (
        max(1, int(round(args.audio_prompt_seconds * frame_rate)))
        if args.continuation
        else 0
    )
    encoded_prompt_seconds = (
        prompt_frames / frame_rate if args.continuation else 0.0
    )
    generated_seconds = (
        args.continuation_seconds if args.continuation else args.duration_seconds
    )
    max_new_tokens = args.max_new_tokens or int(
        math.ceil(generated_seconds * frame_rate)
        + int(getattr(model.model, "num_codebooks", 1) or 1)
    )
    generation_kwargs = {
        "do_sample": True,
        "guidance_scale": args.guidance_scale,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_new_tokens": max_new_tokens,
    }
    sample_rate = int(model.model.audio_sample_rate)

    resolved_dataset = OmegaConf.to_container(
        config.runner.dataset,
        resolve=True,
    )
    if isinstance(resolved_dataset, dict):
        resolved_dataset["subset_dir"] = str(datamodule.subset_dir)
        resolved_dataset["token_dir"] = str(datamodule.token_dir)
        resolved_dataset["manifest_path"] = str(datamodule.manifest_path)
    artist_rows = []
    name_by_key = {
        str(record.get("artist_key") or record.get("artist_id") or record.get("artist_name")):
        str(record.get("artist_name") or "")
        for record in records
    }
    for key, index in sorted(vocabulary.items(), key=lambda item: item[1]):
        artist_rows.append(
            {
                "artist_key": key,
                "artist_index": int(index),
                "artist_name": name_by_key.get(key, key) or key,
            }
        )
    run_metadata = {
        "checkpoint_path": str(checkpoint),
        "config_path": str(config_path),
        "dataset": resolved_dataset,
        "num_artists": len(vocabulary),
        "groups_per_artist": args.groups_per_artist,
        "max_artists": args.max_artists,
        "duration_seconds": output_duration_seconds,
        "generation_mode": generation_mode,
        "audio_prompt_seconds": (
            args.audio_prompt_seconds if args.continuation else 0.0
        ),
        "encoded_audio_prompt_seconds": encoded_prompt_seconds,
        "prompt_token_frames": prompt_frames,
        "continuation_seconds": (
            args.continuation_seconds if args.continuation else None
        ),
        "saved_audio_contains_prompt": False,
        "sample_rate": sample_rate,
        "generation_kwargs": generation_kwargs,
        "seed": args.seed,
        "sample_types": list(SAMPLE_TYPES),
        "artists": artist_rows,
    }
    metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Checkpoint config: {config_path}")
    print(f"Artists in checkpoint vocabulary: {len(vocabulary)}")
    print(f"Pending groups: {len(jobs)}; completed groups: {len(completed)}")
    if args.continuation:
        print(
            f"Generation mode: continuation "
            f"({args.audio_prompt_seconds:.2f}s prompt + "
            f"{args.continuation_seconds:.2f}s saved continuation)"
        )
    else:
        print(f"Generation mode: text ({args.duration_seconds:.2f}s)")
    print(
        f"Saved audio duration: {output_duration_seconds:.2f}s; "
        f"max_new_tokens: {max_new_tokens}"
    )
    print(f"Output: {output_dir}")
    if not jobs:
        return

    import soundfile as sf

    with manifest_path.open("a", encoding="utf-8", buffering=1) as manifest:
        for start in tqdm(range(0, len(jobs), args.batch_size), desc="Generating trios"):
            chunk = jobs[start : start + args.batch_size]
            samples = [dataset[job.dataset_index] for job in chunk]
            batch = _move_batch(datamodule._collator(samples), device)
            if args.continuation:
                batch = _with_audio_continuation_prompt(
                    batch,
                    prompt_frames=prompt_frames,
                )
            target_ids = torch.tensor(
                [job.artist_index for job in chunk], device=device, dtype=torch.long
            )
            other_ids = torch.tensor(
                [job.other_artist_index for job in chunk], device=device, dtype=torch.long
            )
            self_condition = model.concept_learner.prepare_condition(
                concept_ids=target_ids,
                direction=-1.0,
            )
            other_condition = model.concept_learner.prepare_condition(
                concept_ids=other_ids,
                direction=-1.0,
            )
            seed = _batch_seed(args.seed, [job.group_id for job in chunk])
            with _autocast_context(device, args.precision):
                _seed_generation(seed, device)
                default_audio = model._generate_with_condition(
                    batch, generation_kwargs, None
                )
                _seed_generation(seed, device)
                self_audio = model._generate_with_condition(
                    batch, generation_kwargs, self_condition
                )
                _seed_generation(seed, device)
                other_audio = model._generate_with_condition(
                    batch, generation_kwargs, other_condition
                )
            generated_by_type = {
                "default": default_audio,
                "suppress_self": self_audio,
                "suppress_other": other_audio,
            }
            if args.continuation:
                audio_by_type = {
                    sample_type: _only_generated_continuation(
                        audio,
                        sample_rate=sample_rate,
                        prompt_seconds=encoded_prompt_seconds,
                        continuation_seconds=args.continuation_seconds,
                    )
                    for sample_type, audio in generated_by_type.items()
                }
            else:
                audio_by_type = {
                    sample_type: fixed_audio_duration(
                        audio,
                        sample_rate,
                        args.duration_seconds,
                    )
                    for sample_type, audio in generated_by_type.items()
                }
            for row_index, (job, sample) in enumerate(zip(chunk, samples)):
                rows = manifest_rows_for_job(
                    job,
                    text=str(sample["text"]),
                    source_metadata=sample["metadata"],
                    sample_rate=sample_rate,
                    duration_seconds=output_duration_seconds,
                    batch_seed=seed,
                )
                for row in rows:
                    row.update(
                        {
                            "generation_mode": generation_mode,
                            "audio_prompt_seconds": (
                                args.audio_prompt_seconds
                                if args.continuation
                                else 0.0
                            ),
                            "encoded_audio_prompt_seconds": (
                                encoded_prompt_seconds
                            ),
                            "continuation_seconds": (
                                args.continuation_seconds
                                if args.continuation
                                else None
                            ),
                            "prompt_token_frames": prompt_frames,
                            "saved_audio_contains_prompt": False,
                        }
                    )
                    destination = output_dir / str(row["audio"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    waveform = audio_by_type[str(row["sample_type"])][
                        row_index
                    ].transpose(0, 1).numpy()
                    sf.write(
                        destination,
                        waveform,
                        sample_rate,
                        format="WAV",
                        subtype="PCM_16",
                    )
                for row in rows:
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()


if __name__ == "__main__":
    main()
