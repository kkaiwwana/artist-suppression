"""JamendoMaxCaps token dataset for MusicGen and explicit artist control.

The dataset consumes the output of ``scripts/convert_subset_to_encodec.py``.
Each caption row points into a uint16 shard through ``token_shard``,
``token_offset`` and ``token_length``.  Shards are cached per DataLoader worker
and codes are converted to ``torch.long`` only by the batch collator.

The Dataset deliberately returns raw caption text rather than precomputed text
embeddings.  The collator tokenizes captions for MusicGen's native text
conditioning; artist controls are selected explicitly from dataset labels.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_CONCEPT_GRANULARITIES = {"artist", "genre"}


def _artist_key(record: Mapping[str, Any]) -> str:
    for field in ("artist_key", "artist_id", "artist_name"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("manifest record has no artist_key, artist_id, or artist_name")


def _track_key(record: Mapping[str, Any]) -> str:
    track_id = record.get("track_id")
    if track_id is not None and str(track_id).strip():
        return f"{_artist_key(record)}::{track_id}"
    audio = record.get("audio")
    if not isinstance(audio, str) or not audio:
        raise ValueError("manifest record has neither track_id nor audio")
    return f"{_artist_key(record)}::{Path(audio).parent.as_posix()}"


def _record_genres(
    record: Mapping[str, Any], genre_field: str = "coarse_genres"
) -> list[str]:
    value = record.get(genre_field)
    if value is None and genre_field == "coarse_genres":
        value = record.get("coarse_genre")
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    return sorted({str(item).strip() for item in values if str(item).strip()})


def load_token_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate a converter token manifest."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"token manifest not found: {manifest_path}. If EnCodec conversion "
            "is still running, wait for it to write metadata/manifest.jsonl."
        )
    records: list[dict[str, Any]] = []
    required = {"audio", "text", "token_shard", "token_offset", "token_length"}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {manifest_path}:{line_number}"
                ) from error
            missing = required.difference(record)
            if missing:
                raise ValueError(
                    f"{manifest_path}:{line_number} is missing {sorted(missing)}"
                )
            if int(record["token_length"]) <= 0 or int(record["token_offset"]) < 0:
                raise ValueError(
                    f"invalid token offset/length at {manifest_path}:{line_number}"
                )
            _artist_key(record)
            records.append(record)
    if not records:
        raise ValueError(f"token manifest contains no usable records: {manifest_path}")
    return records


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def select_records(
    records: Sequence[Mapping[str, Any]],
    *,
    artist_keys: Sequence[str] | None = None,
    genres: Sequence[str] | None = None,
    genre_field: str = "coarse_genres",
    max_artists: int | None = None,
    max_songs_per_artist: int | None = None,
    max_clips_per_song: int | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build a deterministic sub-subset from an already selected subset.

    Explicit artist/genre filters are applied first.  Optional artist, song,
    clip, and total-sample caps then provide inexpensive build-time experiments
    without regenerating or re-encoding the external subset.
    """

    positive_options = {
        "max_artists": max_artists,
        "max_songs_per_artist": max_songs_per_artist,
        "max_clips_per_song": max_clips_per_song,
        "max_samples": max_samples,
    }
    for name, value in positive_options.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when supplied")

    allowed_artists = {str(value) for value in artist_keys or ()}
    allowed_genres = {str(value) for value in genres or ()}
    filtered = [
        dict(record)
        for record in records
        if (not allowed_artists or _artist_key(record) in allowed_artists)
        and (
            not allowed_genres
            or allowed_genres.intersection(_record_genres(record, genre_field))
        )
    ]
    if not filtered:
        raise ValueError("record filters removed every sample")

    if max_artists is not None:
        artists = sorted({_artist_key(record) for record in filtered})
        rng = _stable_rng(seed, "artists")
        rng.shuffle(artists)
        selected_artists = set(artists[:max_artists])
        filtered = [
            record for record in filtered if _artist_key(record) in selected_artists
        ]

    if max_songs_per_artist is not None:
        artist_tracks: dict[str, list[str]] = defaultdict(list)
        for record in filtered:
            artist = _artist_key(record)
            track = _track_key(record)
            if track not in artist_tracks[artist]:
                artist_tracks[artist].append(track)
        retained_tracks: set[str] = set()
        for artist, tracks in artist_tracks.items():
            tracks = sorted(tracks)
            _stable_rng(seed, f"tracks:{artist}").shuffle(tracks)
            retained_tracks.update(tracks[:max_songs_per_artist])
        filtered = [record for record in filtered if _track_key(record) in retained_tracks]

    if max_clips_per_song is not None:
        by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in filtered:
            by_track[_track_key(record)].append(record)
        retained_ids: set[int] = set()
        for track, track_records in by_track.items():
            ordered = sorted(
                track_records,
                key=lambda row: (float(row.get("start_time", 0.0)), row["audio"]),
            )
            if len(ordered) > max_clips_per_song:
                indices = list(range(len(ordered)))
                _stable_rng(seed, f"clips:{track}").shuffle(indices)
                indices = sorted(indices[:max_clips_per_song])
                ordered = [ordered[index] for index in indices]
            retained_ids.update(id(record) for record in ordered)
        filtered = [record for record in filtered if id(record) in retained_ids]

    if max_samples is not None and len(filtered) > max_samples:
        indices = list(range(len(filtered)))
        _stable_rng(seed, "samples").shuffle(indices)
        retained_indices = set(indices[:max_samples])
        filtered = [
            record for index, record in enumerate(filtered) if index in retained_indices
        ]
    if not filtered:
        raise ValueError("build-time subsetting removed every sample")
    return filtered


def _split_counts(
    track_count: int, val_fraction: float, test_fraction: float
) -> tuple[int, int]:
    if track_count <= 1:
        return 0, 0
    val_count = int(round(track_count * val_fraction))
    test_count = int(round(track_count * test_fraction))
    if val_fraction > 0 and val_count == 0 and track_count >= 3:
        val_count = 1
    if test_fraction > 0 and test_count == 0 and track_count >= 3:
        test_count = 1
    while val_count + test_count >= track_count:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
    return val_count, test_count


def split_records_by_track(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Artist-stratified track split that prevents clip leakage."""

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions must be non-negative and sum to < 1")
    tracks_by_artist: dict[str, set[str]] = defaultdict(set)
    for record in records:
        tracks_by_artist[_artist_key(record)].add(_track_key(record))

    assignment: dict[str, str] = {}
    for artist, track_set in tracks_by_artist.items():
        tracks = sorted(track_set)
        _stable_rng(seed, f"split:{artist}").shuffle(tracks)
        val_count, test_count = _split_counts(
            len(tracks), val_fraction, test_fraction
        )
        for track in tracks[:val_count]:
            assignment[track] = "val"
        for track in tracks[val_count : val_count + test_count]:
            assignment[track] = "test"
        for track in tracks[val_count + test_count :]:
            assignment[track] = "train"

    splits = {"train": [], "val": [], "test": []}
    for record in records:
        splits[assignment[_track_key(record)]].append(dict(record))
    if not splits["train"]:
        raise ValueError("split produced an empty training set")
    return splits


@dataclass(frozen=True)
class LabelVocabulary:
    artist_to_index: dict[str, int]
    genre_to_index: dict[str, int]
    genre_field: str = "coarse_genres"

    @property
    def index_to_artist(self) -> list[str]:
        return [
            label
            for label, _ in sorted(self.artist_to_index.items(), key=lambda item: item[1])
        ]

    @property
    def index_to_genre(self) -> list[str]:
        return [
            label
            for label, _ in sorted(self.genre_to_index.items(), key=lambda item: item[1])
        ]


def build_label_vocabulary(
    records: Sequence[Mapping[str, Any]],
    *,
    genre_field: str = "coarse_genres",
) -> LabelVocabulary:
    artists = sorted({_artist_key(record) for record in records})
    genres = sorted(
        {
            genre
            for record in records
            for genre in _record_genres(record, genre_field)
        }
    )
    if not artists:
        raise ValueError("cannot build an empty artist vocabulary")
    if not genres:
        raise ValueError(
            f"cannot build genre vocabulary: field {genre_field!r} is empty"
        )
    return LabelVocabulary(
        artist_to_index={label: index for index, label in enumerate(artists)},
        genre_to_index={label: index for index, label in enumerate(genres)},
        genre_field=genre_field,
    )


def build_artist_genre_matrix(
    records: Sequence[Mapping[str, Any]],
    vocabulary: LabelVocabulary,
    *,
    genre_field: str = "coarse_genres",
) -> torch.Tensor:
    """Return track-balanced artist-to-genre memberships.

    A long song can yield many 30-second clips. Counting every clip would make
    track duration define the artist's genre prior, so each
    ``(artist, track, genre)`` tuple contributes at most once.
    """

    matrix = torch.zeros(
        len(vocabulary.artist_to_index),
        len(vocabulary.genre_to_index),
        dtype=torch.float32,
    )
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        artist = _artist_key(record)
        track = _track_key(record)
        for genre in _record_genres(record, genre_field):
            key = (artist, track, genre)
            genre_index = vocabulary.genre_to_index.get(genre)
            if key in seen or genre_index is None:
                continue
            seen.add(key)
            matrix[vocabulary.artist_to_index[artist], genre_index] += 1.0
    totals = matrix.sum(dim=-1, keepdim=True)
    return torch.where(totals > 0, matrix / totals.clamp_min(1.0), matrix)


class JamendoMaxCapsDataset(Dataset[dict[str, Any]]):
    """Map caption rows to cached EnCodec tokens and concept labels."""

    def __init__(
        self,
        token_dir: str | Path,
        *,
        records: Sequence[Mapping[str, Any]] | None = None,
        manifest_path: str | Path | None = None,
        vocabulary: LabelVocabulary | None = None,
        concept_granularity: str = "artist",
        genre_field: str = "coarse_genres",
        max_cached_shards: int = 2,
        validate_tokens: bool = True,
    ) -> None:
        super().__init__()
        if concept_granularity not in SUPPORTED_CONCEPT_GRANULARITIES:
            raise ValueError(
                f"concept_granularity must be one of "
                f"{SUPPORTED_CONCEPT_GRANULARITIES}"
            )
        if max_cached_shards <= 0:
            raise ValueError("max_cached_shards must be positive")
        self.token_dir = Path(token_dir).expanduser().resolve()
        path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else self.token_dir / "metadata" / "manifest.jsonl"
        )
        self.records = (
            [dict(record) for record in records]
            if records is not None
            else load_token_manifest(path)
        )
        if not self.records:
            raise ValueError("JamendoMaxCapsDataset requires at least one record")
        self.vocabulary = vocabulary or build_label_vocabulary(
            self.records, genre_field=genre_field
        )
        self.artist_labels = [
            self.vocabulary.artist_to_index[_artist_key(record)]
            for record in self.records
        ]
        self.concept_granularity = concept_granularity
        self.genre_field = genre_field
        self.max_cached_shards = max_cached_shards
        self.validate_tokens = validate_tokens
        self._shard_cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_artists(self) -> int:
        return len(self.vocabulary.artist_to_index)

    @property
    def num_genres(self) -> int:
        return len(self.vocabulary.genre_to_index)

    @property
    def num_concepts(self) -> int:
        return self.num_artists if self.concept_granularity == "artist" else self.num_genres

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict()
        return state

    def _token_path(self, record: Mapping[str, Any]) -> Path:
        path = (self.token_dir / str(record["token_shard"])).resolve()
        try:
            path.relative_to(self.token_dir)
        except ValueError as error:
            raise ValueError(f"token shard escapes token_dir: {path}") from error
        return path

    def _load_shard(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        cached = self._shard_cache.pop(path, None)
        if cached is not None:
            self._shard_cache[path] = cached
            return cached
        if not path.is_file():
            raise FileNotFoundError(f"token shard not found: {path}")
        with np.load(path, allow_pickle=False) as archive:
            codes = archive["codes"]
            offsets = archive["offsets"]
        if codes.ndim != 2 or offsets.ndim != 1:
            raise ValueError(f"invalid token shard arrays in {path}")
        cached = (codes, offsets)
        self._shard_cache[path] = cached
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        codes, _ = self._load_shard(self._token_path(record))
        offset = int(record["token_offset"])
        length = int(record["token_length"])
        if offset < 0 or length <= 0 or offset + length > codes.shape[1]:
            raise IndexError(
                f"token slice [{offset}:{offset + length}] exceeds shard length "
                f"{codes.shape[1]} for {record['audio']}"
            )
        token_view = codes[:, offset : offset + length]
        expected_codebooks = record.get("token_codebooks")
        if expected_codebooks is not None and token_view.shape[0] != int(
            expected_codebooks
        ):
            raise ValueError(
                f"expected {expected_codebooks} codebooks, got {token_view.shape[0]}"
            )
        if self.validate_tokens:
            codebook_size = int(record.get("token_codebook_size", 2048))
            if token_view.size and int(token_view.max()) >= codebook_size:
                raise ValueError(f"out-of-range token ID for {record['audio']}")

        artist_key = _artist_key(record)
        artist_label = self.vocabulary.artist_to_index[artist_key]
        genre_labels = torch.zeros(self.num_genres, dtype=torch.float32)
        for genre in _record_genres(record, self.genre_field):
            genre_index = self.vocabulary.genre_to_index.get(genre)
            if genre_index is not None:
                genre_labels[genre_index] = 1.0
        concept_targets: torch.Tensor
        if self.concept_granularity == "artist":
            concept_targets = torch.tensor(artist_label, dtype=torch.long)
        else:
            concept_targets = genre_labels.clone()

        return {
            "audio_tokens": torch.from_numpy(token_view),
            "text": str(record["text"]),
            "artist_label": torch.tensor(artist_label, dtype=torch.long),
            "genre_labels": genre_labels,
            "concept_targets": concept_targets,
            "metadata": {
                "audio": record["audio"],
                "track_id": str(record.get("track_id", "")),
                "artist_key": artist_key,
                "artist_name": str(record.get("artist_name", "")),
                "genres": _record_genres(record, self.genre_field),
                "token_shard": record["token_shard"],
                "token_offset": offset,
                "token_length": length,
            },
        }


class JamendoMaxCapsCollator:
    """Pad EnCodec codes and tokenize captions for a MusicGen batch."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_text_length: int = 128,
        max_audio_token_length: int | None = None,
        audio_pad_token_id: int = 2048,
    ) -> None:
        if tokenizer is None:
            raise ValueError("a MusicGen-compatible text tokenizer is required")
        if max_text_length <= 0:
            raise ValueError("max_text_length must be positive")
        if max_audio_token_length is not None and max_audio_token_length <= 0:
            raise ValueError("max_audio_token_length must be positive")
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_audio_token_length = max_audio_token_length
        self.audio_pad_token_id = int(audio_pad_token_id)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        texts = [str(sample["text"]) for sample in samples]
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = torch.as_tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(
            tokenized["attention_mask"], dtype=torch.long
        )

        codebooks = int(samples[0]["audio_tokens"].shape[0])
        lengths = [int(sample["audio_tokens"].shape[-1]) for sample in samples]
        if self.max_audio_token_length is not None:
            lengths = [min(length, self.max_audio_token_length) for length in lengths]
        max_length = max(lengths)
        audio_tokens = torch.full(
            (len(samples), codebooks, max_length),
            self.audio_pad_token_id,
            dtype=torch.long,
        )
        decoder_attention_mask = torch.zeros(
            (len(samples), max_length), dtype=torch.long
        )
        for batch_index, (sample, length) in enumerate(zip(samples, lengths)):
            sample_tokens = sample["audio_tokens"]
            if sample_tokens.ndim != 2 or sample_tokens.shape[0] != codebooks:
                raise ValueError("every sample must have audio_tokens shaped [Q, T]")
            audio_tokens[batch_index, :, :length] = sample_tokens[:, :length].long()
            decoder_attention_mask[batch_index, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_tokens": audio_tokens,
            "decoder_attention_mask": decoder_attention_mask,
            "text": texts,
            "artist_label": torch.stack(
                [torch.as_tensor(sample["artist_label"]).long() for sample in samples]
            ),
            "genre_labels": torch.stack(
                [torch.as_tensor(sample["genre_labels"]).float() for sample in samples]
            ),
            "concept_targets": torch.stack(
                [torch.as_tensor(sample["concept_targets"]) for sample in samples]
            ),
            "metadata": [sample["metadata"] for sample in samples],
        }


__all__ = [
    "JamendoMaxCapsCollator",
    "JamendoMaxCapsDataset",
    "LabelVocabulary",
    "SUPPORTED_CONCEPT_GRANULARITIES",
    "build_artist_genre_matrix",
    "build_label_vocabulary",
    "load_token_manifest",
    "select_records",
    "split_records_by_track",
]
