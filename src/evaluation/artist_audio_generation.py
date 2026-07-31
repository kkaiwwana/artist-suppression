"""Planning and storage helpers for paired artist-suppression generations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence
import unicodedata

import torch
import torch.nn.functional as F
from torch import Tensor

from src.evaluation.suppression_similarity import sample_artist_indices


SAMPLE_TYPES = ("default", "suppress_self", "suppress_other")


@dataclass(frozen=True)
class GenerationJob:
    """One caption paired across the three classifier-evaluation branches."""

    group_id: str
    group_index: int
    dataset_index: int
    artist_key: str
    artist_index: int
    artist_name: str
    artist_directory: str
    other_artist_key: str
    other_artist_index: int
    other_artist_name: str


def safe_artist_directory(artist_index: int, artist_name: str) -> str:
    """Return a stable Windows-safe directory with the class index embedded."""

    ascii_name = unicodedata.normalize("NFKD", str(artist_name)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")
    return f"{int(artist_index):04d}_{slug or 'artist'}"


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_generation_jobs(
    records: Sequence[Mapping[str, Any]],
    artist_to_index: Mapping[str, int],
    *,
    groups_per_artist: int,
    seed: int,
    max_artists: int | None = None,
) -> list[GenerationJob]:
    """Build deterministic per-artist jobs and unmatched suppression controls."""

    if groups_per_artist <= 0:
        raise ValueError("groups_per_artist must be positive")
    artists = sorted(artist_to_index, key=lambda key: artist_to_index[key])
    if max_artists is not None:
        if max_artists <= 0:
            raise ValueError("max_artists must be positive")
        artists = artists[:max_artists]
    if len(artist_to_index) < 2:
        raise ValueError("suppress_other generation requires at least two artists")

    names: dict[str, str] = {}
    for record in records:
        key = str(
            record.get("artist_key")
            or record.get("artist_id")
            or record.get("artist_name")
        )
        if key in artist_to_index and key not in names:
            names[key] = str(record.get("artist_name") or key)

    all_artists = sorted(artist_to_index, key=lambda key: artist_to_index[key])
    jobs: list[GenerationJob] = []
    for artist_key in artists:
        indices = sample_artist_indices(
            records,
            artist_key,
            count=groups_per_artist,
            seed=seed,
        )
        candidates = [key for key in all_artists if key != artist_key]
        artist_index = int(artist_to_index[artist_key])
        artist_name = names.get(artist_key, artist_key)
        for group_index, dataset_index in enumerate(indices):
            other_key = _stable_rng(
                seed, f"other:{artist_key}:{group_index}"
            ).choice(candidates)
            jobs.append(
                GenerationJob(
                    group_id=f"{artist_index:04d}:{group_index:03d}",
                    group_index=group_index,
                    dataset_index=dataset_index,
                    artist_key=artist_key,
                    artist_index=artist_index,
                    artist_name=artist_name,
                    artist_directory=safe_artist_directory(
                        artist_index, artist_name
                    ),
                    other_artist_key=other_key,
                    other_artist_index=int(artist_to_index[other_key]),
                    other_artist_name=names.get(other_key, other_key),
                )
            )
    return jobs


def fixed_audio_duration(audio: Tensor, sample_rate: int, seconds: float) -> Tensor:
    """Trim or right-pad batched audio to an exact storage duration."""

    if seconds <= 0 or sample_rate <= 0:
        raise ValueError("seconds and sample_rate must be positive")
    if audio.ndim == 2:
        audio = audio.unsqueeze(1)
    if audio.ndim != 3:
        raise ValueError("audio must have shape [B,T] or [B,C,T]")
    target = int(round(float(seconds) * int(sample_rate)))
    audio = audio.detach().float().cpu()
    if audio.shape[-1] >= target:
        return audio[..., :target]
    return F.pad(audio, (0, target - audio.shape[-1]))


def load_completed_group_ids(manifest_path: str | Path) -> set[str]:
    """Return completed groups whose three audio files still exist."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        return set()
    root = manifest.parent
    grouped: dict[str, set[str]] = {}
    paths: dict[str, list[Path]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = str(row["group_id"])
            grouped.setdefault(group_id, set()).add(str(row["sample_type"]))
            path = Path(str(row["audio"]))
            paths.setdefault(group_id, []).append(
                path if path.is_absolute() else root / path
            )
    required = set(SAMPLE_TYPES)
    return {
        group_id
        for group_id, types in grouped.items()
        if types == required and all(path.is_file() for path in paths[group_id])
    }


def manifest_rows_for_job(
    job: GenerationJob,
    *,
    text: str,
    source_metadata: Mapping[str, Any],
    sample_rate: int,
    duration_seconds: float,
    batch_seed: int,
) -> list[dict[str, Any]]:
    """Build the three manifest rows corresponding to a saved job."""

    rows = []
    for sample_type in SAMPLE_TYPES:
        condition_key = None
        condition_index = None
        if sample_type == "suppress_self":
            condition_key = job.artist_key
            condition_index = job.artist_index
        elif sample_type == "suppress_other":
            condition_key = job.other_artist_key
            condition_index = job.other_artist_index
        relative_audio = (
            Path(job.artist_directory)
            / sample_type
            / f"group_{job.group_index:03d}.wav"
        ).as_posix()
        rows.append(
            {
                **asdict(job),
                "sample_type": sample_type,
                "audio": relative_audio,
                "text": str(text),
                "condition_artist_key": condition_key,
                "condition_artist_index": condition_index,
                "sample_rate": int(sample_rate),
                "duration": float(duration_seconds),
                "generation_seed": int(batch_seed),
                "source_track_id": str(source_metadata.get("track_id", "")),
                "source_audio": str(source_metadata.get("audio", "")),
            }
        )
    return rows


__all__ = [
    "GenerationJob",
    "SAMPLE_TYPES",
    "build_generation_jobs",
    "fixed_audio_duration",
    "load_completed_group_ids",
    "manifest_rows_for_job",
    "safe_artist_directory",
]
