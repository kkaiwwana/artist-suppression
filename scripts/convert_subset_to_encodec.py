"""Convert a JamendoMaxCaps subset into MusicGen-compatible EnCodec tokens.

The input is a subset directory produced by
``download_jamendomaxcaps_subset.py``.  Its ``metadata/manifest.jsonl`` is
read, duplicate audio paths are encoded only once, and every original caption
row is retained in the token manifest.  Codes are stored as uncompressed
``uint16`` NumPy shards so that training can cast them to ``torch.long`` only
after loading.

The conversion is resumable at shard boundaries.  Re-running the same command
validates the input/configuration and continues after the last complete shard.

Example::

    D:\\conda\\python.exe scripts/convert_subset_to_encodec.py ^
        --subset-dir datasets/jamendo_max_caps/subsets/signature_artists0256_songs064_seed0042 ^
        --device cuda --batch-size 8 --num-workers 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


FORMAT_VERSION = 1
DEFAULT_BANDWIDTH = 2.2
DEFAULT_SHARD_SIZE = 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _path_key(path: Path) -> str:
    """Return a stable, case-insensitive key on Windows."""

    value = str(path.resolve())
    return value.casefold() if os.name == "nt" else value


def _resolve_audio_path(subset_dir: Path, audio: str) -> Path:
    path = Path(audio)
    return path.resolve() if path.is_absolute() else (subset_dir / path).resolve()


@dataclass(frozen=True)
class AudioAsset:
    source_index: int
    audio: str
    path: Path


@dataclass
class EncodedAsset:
    source_index: int
    audio: str
    codes: np.ndarray | None = None
    num_samples: int | None = None
    duration_seconds: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.codes is not None


def discover_assets(
    manifest_path: Path, subset_dir: Path
) -> tuple[list[AudioAsset], int, int]:
    """Read the manifest and return first-seen unique audio assets."""

    assets: list[AudioAsset] = []
    seen: set[str] = set()
    rows = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {manifest_path} at line {line_number}"
                ) from error
            audio = record.get("audio")
            if not isinstance(audio, str) or not audio.strip():
                raise ValueError(
                    f"manifest line {line_number} has no non-empty 'audio' field"
                )
            path = _resolve_audio_path(subset_dir, audio)
            key = _path_key(path)
            if key in seen:
                continue
            seen.add(key)
            assets.append(AudioAsset(len(assets), audio, path))
    if not assets:
        raise ValueError(f"manifest contains no audio rows: {manifest_path}")
    return assets, rows, rows - len(assets)


class SubsetAudioDataset(Dataset[dict[str, Any]]):
    """Load FLAC clips while preserving per-file failures as data."""

    def __init__(
        self,
        assets: Sequence[AudioAsset],
        *,
        sample_rate: int,
        channels: int,
    ) -> None:
        self.assets = list(assets)
        self.sample_rate = sample_rate
        self.channels = channels

    def __len__(self) -> int:
        return len(self.assets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        asset = self.assets[index]
        try:
            audio, sample_rate = sf.read(
                asset.path, dtype="float32", always_2d=True
            )
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"expected {self.sample_rate} Hz, found {sample_rate} Hz"
                )
            if audio.shape[1] != self.channels:
                raise ValueError(
                    f"expected {self.channels} channel(s), found {audio.shape[1]}"
                )
            if audio.shape[0] == 0:
                raise ValueError("audio has no samples")
            # SoundFile returns [samples, channels]; EnCodec consumes [C, T].
            waveform = np.ascontiguousarray(audio.T)
            return {
                "source_index": asset.source_index,
                "audio": asset.audio,
                "waveform": waveform,
                "num_samples": waveform.shape[-1],
                "error": None,
            }
        except Exception as error:  # noqa: BLE001 - record bad input and continue
            return {
                "source_index": asset.source_index,
                "audio": asset.audio,
                "waveform": None,
                "num_samples": None,
                "error": f"{type(error).__name__}: {error}",
            }


def collate_audio(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad valid waveforms and retain the original item order."""

    valid = [item for item in items if item["error"] is None]
    input_values: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None
    valid_positions: list[int] = []
    if valid:
        channels = valid[0]["waveform"].shape[0]
        max_samples = max(item["num_samples"] for item in valid)
        input_values = torch.zeros(
            (len(valid), channels, max_samples), dtype=torch.float32
        )
        padding_mask = torch.zeros(
            (len(valid), channels, max_samples), dtype=torch.bool
        )
        valid_index = 0
        for position, item in enumerate(items):
            waveform = item["waveform"]
            if waveform is None:
                continue
            length = item["num_samples"]
            input_values[valid_index, :, :length] = torch.from_numpy(waveform)
            padding_mask[valid_index, :, :length] = True
            valid_positions.append(position)
            valid_index += 1
    return {
        "items": list(items),
        "valid_positions": valid_positions,
        "input_values": input_values,
        "padding_mask": padding_mask,
    }


def _load_prefixed_safetensors(
    model: torch.nn.Module, model_dir: Path, prefix: str
) -> bool:
    """Load one submodule without materializing the full MusicGen model."""

    try:
        from safetensors import safe_open
    except ImportError:
        return False

    index_path = model_dir / "model.safetensors.index.json"
    files_to_keys: dict[Path, list[str]] = {}
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        for key, filename in weight_map.items():
            if key.startswith(prefix):
                files_to_keys.setdefault(model_dir / filename, []).append(key)
    else:
        weights_path = model_dir / "model.safetensors"
        if not weights_path.is_file():
            return False
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = [key for key in handle.keys() if key.startswith(prefix)]
        files_to_keys[weights_path] = keys

    if not files_to_keys or not any(files_to_keys.values()):
        return False
    state_dict: dict[str, torch.Tensor] = {}
    for weights_path, keys in files_to_keys.items():
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            for key in keys:
                state_dict[key[len(prefix) :]] = handle.get_tensor(key)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"failed to load EnCodec weights: {incompatible}")
    return True


def load_encodec_model(
    model_reference: str,
    *,
    revision: str,
    local_files_only: bool,
) -> tuple[torch.nn.Module, Any, str]:
    """Load standalone EnCodec or extract it efficiently from MusicGen."""

    from transformers import (
        AutoConfig,
        EncodecModel,
        MusicgenForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(
        model_reference,
        revision=revision,
        local_files_only=local_files_only,
    )
    if config.model_type == "encodec":
        model = EncodecModel.from_pretrained(
            model_reference,
            revision=revision,
            local_files_only=local_files_only,
        )
        return model, model.config, model_reference

    if config.model_type != "musicgen":
        raise ValueError(
            "--model must refer to an EnCodecModel or MusicGen checkpoint; "
            f"found model_type={config.model_type!r}"
        )

    codec_config = config.audio_encoder
    local_dir = Path(model_reference).expanduser()
    if local_dir.is_dir():
        codec = EncodecModel(codec_config)
        if _load_prefixed_safetensors(codec, local_dir, "audio_encoder."):
            return codec, codec.config, f"{local_dir.resolve()}#audio_encoder"

    # Fallback for remote MusicGen repositories and legacy PyTorch checkpoints.
    full_model = MusicgenForConditionalGeneration.from_pretrained(
        model_reference,
        revision=revision,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    codec = full_model.audio_encoder
    full_model.audio_encoder = torch.nn.Identity()
    del full_model
    return codec, codec.config, f"{model_reference}#audio_encoder"


def expected_num_codebooks(codec_config: Any, bandwidth: float) -> int:
    bits_per_codebook = int(math.log2(codec_config.codebook_size))
    frame_rate = codec_config.sampling_rate / codec_config.hop_length
    return int(round(bandwidth * 1000 / (frame_rate * bits_per_codebook)))


def encode_collated_batch(
    codec: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    bandwidth: float,
    sample_rate: int,
    hop_length: int,
    codebook_size: int,
    expected_codebooks: int,
) -> list[EncodedAsset]:
    """Encode one collated batch and remove frames created by padding."""

    items = batch["items"]
    results: list[EncodedAsset | None] = [None] * len(items)
    for position, item in enumerate(items):
        if item["error"] is not None:
            results[position] = EncodedAsset(
                item["source_index"], item["audio"], error=item["error"]
            )

    input_values = batch["input_values"]
    if input_values is not None:
        padding_mask = batch["padding_mask"]
        input_values = input_values.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        with torch.inference_mode():
            encoded = codec.encode(
                input_values,
                padding_mask=padding_mask,
                bandwidth=bandwidth,
                return_dict=True,
            )
        # [chunks, B, Q, T] -> [B, Q, chunks*T]. MusicGen 32 kHz has one
        # chunk, but this also handles a chunked standalone EnCodec model.
        audio_codes = encoded.audio_codes
        if audio_codes.ndim != 4:
            raise RuntimeError(
                f"unexpected EnCodec code shape: {tuple(audio_codes.shape)}"
            )
        codes = audio_codes.permute(1, 2, 0, 3).reshape(
            audio_codes.shape[1], audio_codes.shape[2], -1
        )
        last_pad = int(getattr(encoded, "last_frame_pad_length", 0) or 0)
        if last_pad:
            codes = codes[..., :-last_pad]
        if codes.shape[1] != expected_codebooks:
            raise RuntimeError(
                f"expected {expected_codebooks} codebooks, got {codes.shape[1]}"
            )
        if codes.numel() and (
            int(codes.min()) < 0 or int(codes.max()) >= codebook_size
        ):
            raise RuntimeError(
                f"EnCodec ID outside [0, {codebook_size - 1}]"
            )
        codes = codes.to(device="cpu", dtype=torch.int32).numpy()

        for encoded_index, position in enumerate(batch["valid_positions"]):
            item = items[position]
            token_length = min(
                codes.shape[-1], math.ceil(item["num_samples"] / hop_length)
            )
            item_codes = np.asarray(
                codes[encoded_index, :, :token_length], dtype=np.uint16
            ).copy()
            results[position] = EncodedAsset(
                item["source_index"],
                item["audio"],
                codes=item_codes,
                num_samples=item["num_samples"],
                duration_seconds=item["num_samples"] / sample_rate,
            )

    if any(result is None for result in results):
        raise RuntimeError("internal error: collated batch produced missing results")
    return [result for result in results if result is not None]


class ShardWriter:
    """Atomically persist fixed-size groups of source assets."""

    def __init__(
        self,
        output_dir: Path,
        *,
        shard_size: int,
        next_shard: int,
        num_codebooks: int,
    ) -> None:
        self.output_dir = output_dir
        self.shards_dir = output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.next_shard = next_shard
        self.num_codebooks = num_codebooks
        self.pending: list[EncodedAsset] = []

    def add(self, result: EncodedAsset) -> None:
        self.pending.append(result)
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self) -> tuple[Path, int] | None:
        if not self.pending:
            return None
        shard_name = f"encodec-{self.next_shard:05d}"
        token_path = self.shards_dir / f"{shard_name}.npz"
        index_path = self.shards_dir / f"{shard_name}.index.jsonl"
        arrays = [item.codes for item in self.pending if item.ok]
        offsets = [0]
        for array in arrays:
            if array is None or array.shape[0] != self.num_codebooks:
                raise RuntimeError("inconsistent EnCodec codebook count")
            offsets.append(offsets[-1] + array.shape[-1])
        concatenated = (
            np.concatenate(arrays, axis=1)
            if arrays
            else np.empty((self.num_codebooks, 0), dtype=np.uint16)
        )
        concatenated = np.asarray(concatenated, dtype="<u2")
        offset_array = np.asarray(offsets, dtype="<u8")

        temporary = token_path.with_suffix(token_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, codes=concatenated, offsets=offset_array)
        temporary.replace(token_path)

        relative_token_path = token_path.relative_to(self.output_dir).as_posix()
        rows: list[dict[str, Any]] = []
        success_index = 0
        for item in self.pending:
            base: dict[str, Any] = {
                "source_index": item.source_index,
                "audio": item.audio,
            }
            if item.ok:
                base.update(
                    {
                        "status": "ok",
                        "token_shard": relative_token_path,
                        "token_index": success_index,
                        "token_offset": offsets[success_index],
                        "token_length": item.codes.shape[-1],
                        "num_samples": item.num_samples,
                        "duration_seconds": item.duration_seconds,
                    }
                )
                success_index += 1
            else:
                base.update({"status": "error", "error": item.error})
            rows.append(base)
        _atomic_jsonl(index_path, rows)
        processed = len(self.pending)
        self.pending.clear()
        self.next_shard += 1
        return token_path, processed


def scan_completed_shards(
    output_dir: Path, assets: Sequence[AudioAsset]
) -> tuple[list[dict[str, Any]], int]:
    """Validate the resumable prefix and remove only incomplete owned files."""

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    for temporary in shards_dir.glob("*.tmp"):
        temporary.unlink()

    references: list[dict[str, Any]] = []
    shard_number = 0
    while True:
        stem = f"encodec-{shard_number:05d}"
        token_path = shards_dir / f"{stem}.npz"
        index_path = shards_dir / f"{stem}.index.jsonl"
        if not token_path.exists() and not index_path.exists():
            break
        if not token_path.is_file() or not index_path.is_file():
            print(f"Removing incomplete shard {stem} before resuming.", flush=True)
            if token_path.exists():
                token_path.unlink()
            if index_path.exists():
                index_path.unlink()
            break
        with index_path.open("r", encoding="utf-8") as handle:
            shard_rows = [json.loads(line) for line in handle if line.strip()]
        for row in shard_rows:
            expected_index = len(references)
            if row.get("source_index") != expected_index:
                raise RuntimeError(
                    f"non-contiguous resume index in {index_path}: expected "
                    f"{expected_index}, found {row.get('source_index')}"
                )
            if expected_index >= len(assets):
                raise RuntimeError("token cache contains more assets than the manifest")
            if row.get("audio") != assets[expected_index].audio:
                raise RuntimeError(
                    "input manifest order changed; use a new --output-dir"
                )
            references.append(row)
        shard_number += 1
    return references, shard_number


def build_output_metadata(
    *,
    subset_dir: Path,
    input_manifest: Path,
    output_dir: Path,
    assets: Sequence[AudioAsset],
    references: Sequence[Mapping[str, Any]],
    num_codebooks: int,
    codebook_size: int,
) -> dict[str, Any]:
    """Rebuild assets, failures, and caption-level token manifests."""

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(metadata_dir / "assets.jsonl", references)
    failures = [row for row in references if row.get("status") != "ok"]
    _atomic_jsonl(metadata_dir / "failures.jsonl", failures)

    by_path: dict[str, Mapping[str, Any]] = {}
    for asset, reference in zip(assets, references, strict=False):
        by_path[_path_key(asset.path)] = reference

    output_manifest = metadata_dir / "manifest.jsonl"
    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    input_rows = output_rows = failed_rows = unprocessed_rows = 0
    total_token_steps = 0
    with input_manifest.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            input_rows += 1
            record = json.loads(line)
            audio_path = _resolve_audio_path(subset_dir, record["audio"])
            reference = by_path.get(_path_key(audio_path))
            if reference is None:
                unprocessed_rows += 1
                continue
            if reference.get("status") != "ok":
                failed_rows += 1
                continue
            enriched = dict(record)
            enriched.update(
                {
                    "token_shard": reference["token_shard"],
                    "token_offset": reference["token_offset"],
                    "token_length": reference["token_length"],
                    "token_codebooks": num_codebooks,
                    "token_codebook_size": codebook_size,
                    "token_dtype": "uint16",
                }
            )
            destination.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            output_rows += 1
            total_token_steps += int(reference["token_length"])
    temporary.replace(output_manifest)
    return {
        "input_manifest_rows": input_rows,
        "output_manifest_rows": output_rows,
        "failed_manifest_rows": failed_rows,
        "unprocessed_manifest_rows": unprocessed_rows,
        "total_manifest_token_steps": total_token_steps,
    }


def _model_reference(value: str) -> str:
    expanded = Path(value).expanduser()
    return str(expanded.resolve()) if expanded.exists() else value


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-dir",
        type=Path,
        required=True,
        help="Subset containing metadata/manifest.jsonl and clips/",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the input manifest (default: SUBSET/metadata/manifest.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Token cache directory (default: "
            "SUBSET/encodec_musicgen_32khz_4cb)"
        ),
    )
    parser.add_argument(
        "--model",
        default=str(repo_root / "models" / "musicgen-small"),
        help="Standalone EnCodec or MusicGen checkpoint",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disallow Hugging Face downloads",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Process at most the first N unique clips; useful for verification",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.shard_size <= 0:
        raise ValueError("batch-size/shard-size must be positive and workers non-negative")
    if args.max_clips is not None and args.max_clips <= 0:
        raise ValueError("--max-clips must be positive")

    subset_dir = args.subset_dir.expanduser().resolve()
    input_manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else subset_dir / "metadata" / "manifest.jsonl"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else subset_dir / "encodec_musicgen_32khz_4cb"
    )
    if not input_manifest.is_file():
        raise FileNotFoundError(f"input manifest not found: {input_manifest}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subset:  {subset_dir}", flush=True)
    print(f"Manifest: {input_manifest}", flush=True)
    print(f"Output:   {output_dir}", flush=True)
    assets, manifest_rows, duplicate_rows = discover_assets(
        input_manifest, subset_dir
    )
    manifest_sha256 = _sha256(input_manifest)
    target_assets = (
        min(len(assets), args.max_clips)
        if args.max_clips is not None
        else len(assets)
    )
    print(
        f"Found {manifest_rows:,} caption rows and {len(assets):,} unique clips "
        f"({duplicate_rows:,} duplicate audio references).",
        flush=True,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    model_reference = _model_reference(args.model)
    print(f"Loading EnCodec from {model_reference} on {device} ...", flush=True)
    codec, codec_config, resolved_codec = load_encodec_model(
        model_reference,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if codec_config.normalize:
        raise ValueError(
            "this cache format currently requires EnCodec normalize=false; "
            "the standard MusicGen 32 kHz codec satisfies this"
        )
    if args.bandwidth not in codec_config.target_bandwidths:
        raise ValueError(
            f"bandwidth {args.bandwidth} is not supported; choose from "
            f"{codec_config.target_bandwidths}"
        )
    if codec_config.codebook_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("codebook IDs do not fit in uint16")
    num_codebooks = expected_num_codebooks(codec_config, args.bandwidth)
    codec.eval().requires_grad_(False).to(device)

    config_path = output_dir / "metadata" / "config.json"
    stable_config = {
        "format_version": FORMAT_VERSION,
        "subset_dir": str(subset_dir),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": manifest_sha256,
        "model_reference": model_reference,
        "model_revision": args.revision,
        "resolved_codec": resolved_codec,
        "sample_rate": int(codec_config.sampling_rate),
        "channels": int(codec_config.audio_channels),
        "bandwidth_kbps": args.bandwidth,
        "frame_rate_hz": codec_config.sampling_rate / codec_config.hop_length,
        "hop_length": int(codec_config.hop_length),
        "num_codebooks": num_codebooks,
        "codebook_size": int(codec_config.codebook_size),
        "token_dtype": "uint16",
    }
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        mismatches = [
            key
            for key, value in stable_config.items()
            if previous.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "existing token cache is incompatible for fields "
                f"{mismatches}; choose a new --output-dir"
            )
    else:
        _atomic_json(
            config_path,
            {**stable_config, "created_at": _utc_now(), "status": "running"},
        )

    references, next_shard = scan_completed_shards(output_dir, assets)
    if len(references) > target_assets:
        target_assets = len(references)
    print(
        f"Resume point: {len(references):,}/{target_assets:,} unique clips; "
        f"batch={args.batch_size}, shard={args.shard_size}.",
        flush=True,
    )

    writer = ShardWriter(
        output_dir,
        shard_size=args.shard_size,
        next_shard=next_shard,
        num_codebooks=num_codebooks,
    )
    pending_assets = assets[len(references) : target_assets]
    dataset = SubsetAudioDataset(
        pending_assets,
        sample_rate=int(codec_config.sampling_rate),
        channels=int(codec_config.audio_channels),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0 and len(dataset) > 0,
    )

    started = time.monotonic()
    interrupted = False
    progress = tqdm(
        total=target_assets,
        initial=len(references),
        unit="clip",
        dynamic_ncols=True,
        desc="EnCodec",
    )
    try:
        for batch in loader:
            results = encode_collated_batch(
                codec,
                batch,
                device=device,
                bandwidth=args.bandwidth,
                sample_rate=int(codec_config.sampling_rate),
                hop_length=int(codec_config.hop_length),
                codebook_size=int(codec_config.codebook_size),
                expected_codebooks=num_codebooks,
            )
            for result in results:
                writer.add(result)
            progress.update(len(results))
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; saving the completed in-memory batch...", flush=True)
    finally:
        writer.flush()
        progress.close()

    references, _ = scan_completed_shards(output_dir, assets)
    manifest_stats = build_output_metadata(
        subset_dir=subset_dir,
        input_manifest=input_manifest,
        output_dir=output_dir,
        assets=assets,
        references=references,
        num_codebooks=num_codebooks,
        codebook_size=int(codec_config.codebook_size),
    )
    successful = [row for row in references if row.get("status") == "ok"]
    failed = len(references) - len(successful)
    token_steps = sum(int(row["token_length"]) for row in successful)
    shard_bytes = sum(path.stat().st_size for path in (output_dir / "shards").glob("*.npz"))
    complete = len(references) >= len(assets)
    status = "interrupted" if interrupted else "complete" if complete else "partial"
    run_stats = {
        "status": status,
        "updated_at": _utc_now(),
        "elapsed_seconds_this_run": time.monotonic() - started,
        "manifest_rows": manifest_rows,
        "unique_clips": len(assets),
        "duplicate_audio_references": duplicate_rows,
        "processed_unique_clips": len(references),
        "successful_unique_clips": len(successful),
        "failed_unique_clips": failed,
        "token_steps": token_steps,
        "token_ids": token_steps * num_codebooks,
        "token_shard_bytes": shard_bytes,
        **manifest_stats,
    }
    _atomic_json(output_dir / "metadata" / "run_stats.json", run_stats)
    with config_path.open("r", encoding="utf-8") as handle:
        output_config = json.load(handle)
    output_config.update({"status": status, "updated_at": _utc_now()})
    _atomic_json(config_path, output_config)
    print(json.dumps(run_stats, ensure_ascii=False, indent=2), flush=True)

    if interrupted:
        return 130
    if failed:
        print("Some clips failed; see metadata/failures.jsonl.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
