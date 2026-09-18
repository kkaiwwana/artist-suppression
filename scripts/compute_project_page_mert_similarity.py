"""Compute per-demo MERT cosine similarity to each sample's GT reference.

The output is a small static JSON sidecar consumed by the project page. Audio
fingerprints make repeated runs incremental: unchanged samples reuse their
recorded values, while new or modified samples are encoded again.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import soundfile as sf
import torch
from torch import Tensor
import torch.nn.functional as F


METRIC_ID = "mert_similarity_to_reference"
DEFAULT_MODEL = "m-a-p/MERT-v1-95M"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_inside(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Audio path escapes manifest root: {relative_path}") from exc
    return candidate


def audio_fingerprint(path: Path) -> str:
    """Return a content fingerprint used to reuse unchanged similarity values."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_mono(path: Path) -> tuple[Tensor, int]:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = values.mean(axis=1)
    return torch.from_numpy(mono), int(sample_rate)


def encode_audio_files(
    encoder: Any,
    records: Iterable[tuple[tuple[str, str], Path]],
    *,
    batch_size: int,
) -> dict[tuple[str, str], Tensor]:
    """Encode audio in same-length batches and return CPU embeddings by key."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    grouped: dict[tuple[int, int], list[tuple[tuple[str, str], Path]]] = defaultdict(list)
    for key, path in records:
        info = sf.info(path)
        grouped[(int(info.samplerate), int(info.frames))].append((key, path))

    embeddings: dict[tuple[str, str], Tensor] = {}
    for (sample_rate, _frames), group in grouped.items():
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            waveforms = []
            for _key, path in chunk:
                waveform, actual_sample_rate = _load_mono(path)
                if actual_sample_rate != sample_rate:
                    raise ValueError(f"Sample-rate metadata changed while reading {path}")
                waveforms.append(waveform)
            batch = torch.stack(waveforms)
            encoded = torch.as_tensor(
                encoder.encode_audio(batch, sample_rate=sample_rate, batch_size=batch_size)
            ).detach().float().cpu()
            if encoded.ndim != 2 or encoded.shape[0] != len(chunk):
                raise ValueError("MERT encoder must return one [D] embedding per audio file")
            for (key, _path), embedding in zip(chunk, encoded, strict=True):
                embeddings[key] = embedding
    return embeddings


def _model_metadata(model_name_or_path: str, layer: int) -> dict[str, Any]:
    return {
        "id": METRIC_ID,
        "label": "MERT similarity to GT",
        "model": str(model_name_or_path),
        "layer": int(layer),
        "pooling": "attention-mask-aware temporal mean",
        "comparison": "cosine similarity",
        "reference_variant": "reference",
    }


def compute_mert_similarities(
    manifest_path: Path,
    output_path: Path,
    *,
    model_name_or_path: str = DEFAULT_MODEL,
    device: str = "auto",
    layer: int = -1,
    batch_size: int = 8,
    local_files_only: bool = False,
    force: bool = False,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Compute or reuse MERT-to-reference similarities for one web manifest."""

    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    manifest_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = _model_metadata(model_name_or_path, layer)

    existing: dict[str, Any] = {}
    if output_path.is_file() and not force:
        candidate = json.loads(output_path.read_text(encoding="utf-8"))
        if candidate.get("metric") == metadata:
            existing = candidate

    prepared: dict[str, dict[str, Any]] = {}
    reusable: dict[str, dict[str, Any]] = {}
    records_to_encode: list[tuple[tuple[str, str], Path]] = []
    for item in manifest.get("items", []):
        sample_id = str(item["sample_id"])
        variants = item.get("variants", [])
        variant_ids = [str(variant["id"]) for variant in variants]
        if "reference" not in variant_ids:
            raise ValueError(f"{sample_id} has no reference variant")

        paths = {
            str(variant["id"]): _resolve_inside(manifest_root, str(variant["src"]))
            for variant in variants
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        fingerprints = {variant_id: audio_fingerprint(path) for variant_id, path in paths.items()}
        cached = existing.get("items", {}).get(sample_id)
        if (
            cached
            and cached.get("fingerprints") == fingerprints
            and set(cached.get("values", {})) == set(variant_ids)
        ):
            reusable[sample_id] = cached
            continue

        prepared[sample_id] = {
            "item": item,
            "variant_ids": variant_ids,
            "fingerprints": fingerprints,
        }
        records_to_encode.extend(((sample_id, variant_id), path) for variant_id, path in paths.items())

    embeddings: dict[tuple[str, str], Tensor] = {}
    if records_to_encode:
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
            if device == "auto"
            else device
        )
        if encoder is None:
            from src.evaluation.suppression_similarity import MERTEncoder

            encoder = MERTEncoder(
                model_name_or_path,
                device=resolved_device,
                local_files_only=local_files_only,
                layer=layer,
            )
        print(
            f"Encoding {len(records_to_encode)} audio files from {len(prepared)} samples "
            f"on {resolved_device}..."
        )
        embeddings = encode_audio_files(encoder, records_to_encode, batch_size=batch_size)

    output_items: dict[str, Any] = {}
    for item in manifest.get("items", []):
        sample_id = str(item["sample_id"])
        if sample_id in reusable:
            output_items[sample_id] = reusable[sample_id]
            continue

        prepared_item = prepared[sample_id]
        reference = embeddings[(sample_id, "reference")]
        values = {}
        for variant_id in prepared_item["variant_ids"]:
            if variant_id == "reference":
                similarity = 1.0
            else:
                similarity = float(
                    F.cosine_similarity(
                        embeddings[(sample_id, variant_id)].unsqueeze(0),
                        reference.unsqueeze(0),
                        dim=-1,
                    ).item()
                )
            values[variant_id] = round(similarity, 6)
        output_items[sample_id] = {
            "sample_index": item.get("sample_index"),
            "group": item.get("group"),
            "source_title": item.get("source_title"),
            "fingerprints": prepared_item["fingerprints"],
            "values": values,
        }

    result = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metric": metadata,
        "items": output_items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result["summary"] = {
        "items": len(output_items),
        "computed_items": len(prepared),
        "reused_items": len(reusable),
        "audio_files_encoded": len(records_to_encode),
    }
    return result


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "project_page" / "demo_data" / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "project_page" / "demo_data" / "mert_similarity.json",
    )
    parser.add_argument("--model", default=os.environ.get("MERT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = compute_mert_similarities(
        args.manifest,
        args.output,
        model_name_or_path=args.model,
        device=args.device,
        layer=args.layer,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
        force=args.force,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
