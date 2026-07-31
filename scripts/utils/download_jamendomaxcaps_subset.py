"""Build a caption-aligned JamendoMaxCaps subset for MusicGen/EnCodec training.

The script balances artists across 15 coarse genre families, then selects each
artist's most representative songs from a stable genre/instrument/vartag
signature computed over the full candidate catalog.  It downloads source
songs concurrently and uses the time spans in
``final_caption30sec.jsonl`` to create 32 kHz mono FLAC clips plus a JSONL
training manifest.

Example::

    D:\\conda\\python.exe scripts/download_jamendomaxcaps_subset.py ^
        --num-artists 30 --songs-per-artist 10

The default output is::

    datasets/jamendo_max_caps/subsets/signature_artists0030_songs010_seed0042/

The command is resumable: existing downloads and clips are reused.  Source
MP3 files are removed after successful clipping by default; pass
``--keep-originals`` to retain them.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

try:  # orjson is substantially faster for the multi-GB metadata files.
    import orjson
except ImportError:  # pragma: no cover - exercised only in minimal environments
    orjson = None


DATE_METADATA_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")
INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_FORMAT = "s16"
DEFAULT_BITS_PER_SAMPLE = 16
MIN_AUDIO_BYTES = 4_096
MIN_CLIP_BYTES = 1_024
USER_AGENT = "mus-unlearning-jamendomaxcaps-subset/1.0"

# The 220 raw Jamendo genre tags mix broad genres, subgenres, and functional
# production labels.  These 15 families are intentionally coarse: they are
# used only to balance artists across the dataset.  Raw genre/instrument/
# vartags remain available for artist-signature scoring.
COARSE_GENRES = (
    "ambient_newage",
    "blues_funk_soul",
    "cinematic",
    "classical_orchestral",
    "electronic_dance",
    "experimental",
    "folk_country",
    "hiphop_rnb",
    "jazz",
    "latin",
    "metal",
    "pop",
    "reggae_caribbean",
    "rock_alternative_punk",
    "world_traditional",
)


# Exact and substring anchors cover the stable vocabulary.  Remaining raw
# genres are assigned from their co-occurrence with these anchors in the full
# local metadata, rather than being hard-coded from the current sample.
COARSE_GENRE_EXACT: dict[str, str] = {
    "8bit": "electronic_dance",
    "adultcontemporary": "pop",
    "americana": "folk_country",
    "americannative": "world_traditional",
    "asian": "world_traditional",
    "bachata": "latin",
    "balkan": "world_traditional",
    "batucada": "latin",
    "bolero": "latin",
    "bossanova": "latin",
    "cabaret": "cinematic",
    "calypso": "reggae_caribbean",
    "celtic": "world_traditional",
    "chansonfrancaise": "pop",
    "christian": "pop",
    "cumbia": "latin",
    "dancehall": "reggae_caribbean",
    "dub": "reggae_caribbean",
    "easylistening": "pop",
    "fado": "world_traditional",
    "fanfare": "classical_orchestral",
    "flamenco": "latin",
    "folklore": "world_traditional",
    "garage": "rock_alternative_punk",
    "gospel": "blues_funk_soul",
    "gothic": "rock_alternative_punk",
    "gypsy": "world_traditional",
    "indian": "world_traditional",
    "indie": "rock_alternative_punk",
    "jingle": "cinematic",
    "kidsquirky": "cinematic",
    "klezmer": "world_traditional",
    "lofi": "electronic_dance",
    "mambo": "latin",
    "manouche": "jazz",
    "march": "classical_orchestral",
    "mariachi": "latin",
    "medieval": "world_traditional",
    "merengue": "latin",
    "middleeastern": "world_traditional",
    "oriental": "world_traditional",
    "ragga": "reggae_caribbean",
    "ragtime": "jazz",
    "rai": "world_traditional",
    "reggaeton": "latin",
    "rocksteady": "reggae_caribbean",
    "rumba": "latin",
    "salsa": "latin",
    "samba": "latin",
    "ska": "reggae_caribbean",
    "spokenword": "experimental",
    "swing": "jazz",
    "tango": "latin",
    "tribal": "world_traditional",
    "waltz": "classical_orchestral",
    "western": "folk_country",
    "world": "world_traditional",
    "zouk": "reggae_caribbean",
}

COARSE_GENRE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("metal", ("metal", "deathcore", "grindcore", "sludge", "stoner")),
    ("hiphop_rnb", ("hiphop", "rap", "rnb", "grime", "trap", "chillhop")),
    ("jazz", ("jazz", "bebop")),
    ("blues_funk_soul", ("blues", "funk", "soul")),
    (
        "classical_orchestral",
        ("classical", "baroque", "chamber", "choral", "opera", "symphonic", "piano"),
    ),
    ("folk_country", ("folk", "country", "bluegrass", "singersongwriter")),
    (
        "reggae_caribbean",
        ("reggae", "rocksteady", "dancehall", "ragga", "calypso", "ska"),
    ),
    (
        "world_traditional",
        ("afric", "afro", "asian", "indian", "oriental", "tribal", "world", "celtic"),
    ),
    (
        "cinematic",
        ("filmscore", "corporate", "production", "trailer", "musicbed", "jingle", "stinger", "logo", "intro", "loop"),
    ),
    (
        "ambient_newage",
        ("ambient", "newage", "chillout", "downtempo", "drone"),
    ),
    (
        "experimental",
        ("experimental", "avantgarde", "industrial", "noise", "glitch", "spokenword"),
    ),
    (
        "electronic_dance",
        ("electro", "electronic", "techno", "house", "trance", "dance", "edm", "synth", "wave", "dubstep", "drumnbass", "break", "jungle", "idm"),
    ),
    (
        "rock_alternative_punk",
        ("rock", "punk", "grunge", "emo", "shoegaze", "hardcore", "indie"),
    ),
    ("latin", ("latin", "bachata", "bossa", "cumbia", "flamenco", "salsa", "samba", "tango")),
    ("pop", ("pop", "easylistening")),
)


@dataclass(frozen=True)
class Track:
    """Compact metadata retained for an eligible Jamendo track."""

    track_id: str
    title: str | None
    artist_key: str
    artist_id: str | None
    artist_name: str | None
    album_id: str | None
    album_name: str | None
    duration: float | None
    releasedate: str | None
    audiodownload: str
    genres: tuple[str, ...]
    coarse_genres: tuple[str, ...]
    style_tags: tuple[str, ...]


@dataclass(frozen=True)
class Caption:
    """One caption and its exact audio time span."""

    start_time: float
    end_time: float
    text: str

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class SelectedSong:
    """A selected track, its balancing label, and usable captions."""

    track: Track
    balance_genre: str
    captions: tuple[Caption, ...]
    signature_score: float
    signature_tags: tuple[str, ...]


@dataclass(frozen=True)
class ArtistProfile:
    """Stable artist signature computed from the artist's full candidate catalog."""

    dominant_genre: str
    signature_tags: tuple[str, ...]
    signature_weights: tuple[tuple[str, float], ...]
    signature_supports: tuple[tuple[str, float], ...]

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.signature_weights)


@dataclass(frozen=True)
class DownloadResult:
    track_id: str
    path: Path
    status: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ClipTask:
    song: SelectedSong
    caption: Caption
    source_path: Path
    output_path: Path


@dataclass(frozen=True)
class ClipResult:
    task: ClipTask
    status: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _loads(line: bytes | str) -> Any:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_genres(value: Any) -> tuple[str, ...]:
    """Return stable, lowercase, non-empty genre labels."""

    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        values = []
    normalized = {
        str(genre).strip().casefold()
        for genre in values
        if genre is not None and str(genre).strip()
    }
    return tuple(sorted(normalized))


def namespaced_tags(namespace: str, value: Any) -> tuple[str, ...]:
    """Normalize metadata tags while preserving their semantic namespace."""

    return tuple(f"{namespace}:{tag}" for tag in normalize_genres(value))


def safe_path_component(value: str, *, fallback_prefix: str = "item") -> str:
    """Convert an identifier to a short Windows-safe path component."""

    cleaned = INVALID_PATH_CHARS_RE.sub("_", value).strip(" ._")
    if not cleaned:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
        cleaned = f"{fallback_prefix}_{digest}"
    if len(cleaned) > 96:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
        cleaned = f"{cleaned[:80]}_{digest}"
    return cleaned


def artist_directory(track: Track) -> str:
    if track.artist_id:
        return f"artist_{safe_path_component(track.artist_id, fallback_prefix='id')}"
    digest = hashlib.sha1(track.artist_key.encode("utf-8")).hexdigest()[:12]
    return f"artist_name_{digest}"


def discover_metadata_files(metadata_dir: str | Path) -> list[Path]:
    """Find date-partitioned metadata files, excluding the large caption file."""

    directory = Path(metadata_dir).expanduser().resolve()
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and DATE_METADATA_RE.fullmatch(path.name)
    )
    if not files:
        raise FileNotFoundError(
            f"no date-formatted JamendoMaxCaps metadata files found in {directory}"
        )
    return files


def scan_metadata(
    metadata_files: Sequence[str | Path],
) -> tuple[dict[str, list[Track]], Counter[str]]:
    """Stream metadata and group genre-tagged, downloadable tracks by artist."""

    artist_tracks: dict[str, list[Track]] = defaultdict(list)
    seen_track_ids: set[str] = set()
    stats: Counter[str] = Counter()

    for path_like in tqdm(metadata_files, desc="Scanning metadata files", unit="file"):
        path = Path(path_like)
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["metadata_rows"] += 1
                try:
                    item = _loads(line)
                except (json.JSONDecodeError, ValueError, TypeError):
                    stats["metadata_parse_errors"] += 1
                    continue

                track_id = _clean_text(item.get("id"))
                artist_id = _clean_text(item.get("artist_id"))
                artist_name = _clean_text(item.get("artist_name"))
                download_url = _clean_text(item.get("audiodownload"))
                music_info = item.get("musicinfo") or item.get("music_info") or {}
                tags = music_info.get("tags") or {}
                genres = normalize_genres(tags.get("genres"))
                style_tags = tuple(
                    sorted(
                        set(namespaced_tags("genre", genres))
                        | set(namespaced_tags("instrument", tags.get("instruments")))
                        | set(namespaced_tags("vartag", tags.get("vartags")))
                    )
                )

                if not track_id:
                    stats["missing_track_id"] += 1
                    continue
                if track_id in seen_track_ids:
                    stats["duplicate_track_id"] += 1
                    continue
                if not artist_id and not artist_name:
                    stats["missing_artist"] += 1
                    continue
                if not download_url or item.get("audiodownload_allowed", True) is False:
                    stats["not_downloadable"] += 1
                    continue
                if not genres:
                    stats["empty_genres"] += 1
                    continue

                artist_key = (
                    f"id:{artist_id}"
                    if artist_id
                    else f"name:{artist_name.casefold()}"
                )
                track = Track(
                    track_id=track_id,
                    title=_clean_text(item.get("name")),
                    artist_key=artist_key,
                    artist_id=artist_id,
                    artist_name=artist_name,
                    album_id=_clean_text(item.get("album_id")),
                    album_name=_clean_text(item.get("album_name")),
                    duration=_clean_float(item.get("duration")),
                    releasedate=_clean_text(item.get("releasedate")),
                    audiodownload=download_url,
                    genres=genres,
                    coarse_genres=(),
                    style_tags=style_tags,
                )
                seen_track_ids.add(track_id)
                artist_tracks[artist_key].append(track)
                stats["eligible_tracks"] += 1

    stats["artists_with_eligible_tracks"] = len(artist_tracks)
    return dict(artist_tracks), stats


def direct_coarse_genre(raw_genre: str) -> str | None:
    """Map a known genre tag to one of the 15 stable coarse families."""

    if raw_genre in COARSE_GENRE_EXACT:
        return COARSE_GENRE_EXACT[raw_genre]
    for coarse_genre, keywords in COARSE_GENRE_KEYWORDS:
        if any(keyword in raw_genre for keyword in keywords):
            return coarse_genre
    return None


def build_coarse_genre_taxonomy(
    artist_tracks: Mapping[str, Sequence[Track]],
    *,
    min_cooccurrence: int = 5,
    min_confidence: float = 0.35,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Infer a complete raw-to-coarse genre map from full-data co-occurrence.

    Stable keyword anchors provide semantic meaning.  Any remaining tag is
    assigned to the coarse family with which it co-occurs most often, provided
    there is enough evidence.  Unresolved tags are excluded from balancing but
    remain in the raw metadata and artist signature.
    """

    track_frequency: Counter[str] = Counter()
    cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
    direct_mapping: dict[str, str] = {}
    for tracks in artist_tracks.values():
        for track in tracks:
            track_frequency.update(track.genres)
            direct_categories = {
                category
                for genre in track.genres
                if (category := direct_coarse_genre(genre)) is not None
            }
            for genre in track.genres:
                category = direct_coarse_genre(genre)
                if category is not None:
                    direct_mapping[genre] = category
                    continue
                if direct_categories:
                    weight = 1.0 / len(direct_categories)
                    for direct_category in direct_categories:
                        cooccurrence[genre][direct_category] += weight

    mapping = dict(direct_mapping)
    inferred_confidence: dict[str, float] = {}
    unresolved: list[str] = []
    for genre in sorted(track_frequency):
        if genre in mapping:
            continue
        evidence = cooccurrence.get(genre, Counter())
        total = sum(evidence.values())
        if evidence and total >= min_cooccurrence:
            category, votes = max(evidence.items(), key=lambda item: (item[1], item[0]))
            confidence = votes / total
            if confidence >= min_confidence:
                mapping[genre] = category
                inferred_confidence[genre] = confidence
                continue
        unresolved.append(genre)

    taxonomy_stats = {
        "coarse_genres": list(COARSE_GENRES),
        "raw_genre_count": len(track_frequency),
        "directly_mapped_raw_genres": len(direct_mapping),
        "cooccurrence_inferred_raw_genres": len(inferred_confidence),
        "unresolved_raw_genres": unresolved,
        "raw_genre_track_frequency": dict(track_frequency.most_common()),
        "raw_to_coarse": dict(sorted(mapping.items())),
        "inferred_confidence": dict(sorted(inferred_confidence.items())),
    }
    return mapping, taxonomy_stats


def apply_coarse_genre_taxonomy(
    artist_tracks: Mapping[str, Sequence[Track]],
    mapping: Mapping[str, str],
) -> tuple[dict[str, list[Track]], int]:
    """Attach coarse genres and drop tracks with no resolvable genre family."""

    result: dict[str, list[Track]] = {}
    dropped = 0
    for artist_key, tracks in artist_tracks.items():
        mapped_tracks = []
        for track in tracks:
            coarse_genres = tuple(
                sorted({mapping[g] for g in track.genres if g in mapping})
            )
            if not coarse_genres:
                dropped += 1
                continue
            mapped_tracks.append(replace(track, coarse_genres=coarse_genres))
        if mapped_tracks:
            result[artist_key] = mapped_tracks
    return result, dropped


def dominant_coarse_genre(tracks: Sequence[Track]) -> str:
    """Return an artist's dominant coarse family using fractional track votes."""

    votes: Counter[str] = Counter()
    for track in tracks:
        if not track.coarse_genres:
            continue
        weight = 1.0 / len(track.coarse_genres)
        for genre in track.coarse_genres:
            votes[genre] += weight
    if not votes:
        raise ValueError("cannot determine dominant genre without coarse genres")
    order = {genre: index for index, genre in enumerate(COARSE_GENRES)}
    return max(votes, key=lambda genre: (votes[genre], -order[genre]))


def build_artist_profiles(
    artist_tracks: Mapping[str, Sequence[Track]],
    *,
    signature_min_support: float = 0.30,
    signature_min_songs: int = 3,
    max_signature_tags: int = 5,
    min_global_style_tag_count: int = 50,
    signature_idf_cap: float = 3.0,
) -> dict[str, ArtistProfile]:
    """Build stable, distinctive signatures from genre/instrument/vartags."""

    if not 0 < signature_min_support <= 1:
        raise ValueError("signature_min_support must be in (0, 1]")
    if signature_min_songs < 1 or max_signature_tags < 1:
        raise ValueError("signature tag count parameters must be positive")

    global_track_frequency: Counter[str] = Counter()
    global_artist_frequency: Counter[str] = Counter()
    for tracks in artist_tracks.values():
        artist_tags: set[str] = set()
        for track in tracks:
            global_track_frequency.update(set(track.style_tags))
            artist_tags.update(track.style_tags)
        global_artist_frequency.update(artist_tags)

    artist_count = len(artist_tracks)
    profiles: dict[str, ArtistProfile] = {}
    for artist_key, tracks in artist_tracks.items():
        tag_counts: Counter[str] = Counter(
            tag for track in tracks for tag in set(track.style_tags)
        )
        catalog_size = len(tracks)
        stable_tags = [
            tag
            for tag, count in tag_counts.items()
            if count >= signature_min_songs
            and count / catalog_size >= signature_min_support
            and global_track_frequency[tag] >= min_global_style_tag_count
        ]
        # Small synthetic datasets and unusual artists may have no tag passing
        # the global threshold.  Fall back to locally stable tags, never to a
        # one-off tag unless no repeated tag exists at all.
        if not stable_tags:
            stable_tags = [
                tag
                for tag, count in tag_counts.items()
                if count >= min(signature_min_songs, max(2, catalog_size))
                and count / catalog_size >= signature_min_support
            ]
        if not stable_tags:
            stable_tags = [tag for tag, _ in tag_counts.most_common(max_signature_tags)]

        weighted: list[tuple[str, float, float]] = []
        for tag in stable_tags:
            support = tag_counts[tag] / catalog_size
            idf = 1.0 + math.log(
                (artist_count + 1) / (global_artist_frequency[tag] + 1)
            )
            weight = support * min(idf, signature_idf_cap)
            weighted.append((tag, weight, support))
        weighted.sort(key=lambda item: (-item[1], -item[2], item[0]))

        chosen = weighted[:max_signature_tags]
        # Always retain the strongest raw genre when the top list would
        # otherwise contain only instruments/vartags.  It may be below the
        # general support threshold for stylistically broad artists, but is
        # still useful together with the dominant coarse-genre constraint.
        if chosen and not any(tag.startswith("genre:") for tag, _, _ in chosen):
            genre_candidates = []
            for tag, count in tag_counts.items():
                if not tag.startswith("genre:"):
                    continue
                support = count / catalog_size
                idf = 1.0 + math.log(
                    (artist_count + 1) / (global_artist_frequency[tag] + 1)
                )
                genre_candidates.append(
                    (tag, support * min(idf, signature_idf_cap), support)
                )
            best_genre = (
                max(genre_candidates, key=lambda item: (item[1], item[2], item[0]))
                if genre_candidates
                else None
            )
            if best_genre is not None:
                chosen[-1] = best_genre
                chosen.sort(key=lambda item: (-item[1], -item[2], item[0]))

        profiles[artist_key] = ArtistProfile(
            dominant_genre=dominant_coarse_genre(tracks),
            signature_tags=tuple(item[0] for item in chosen),
            signature_weights=tuple((item[0], item[1]) for item in chosen),
            signature_supports=tuple((item[0], item[2]) for item in chosen),
        )
    return profiles


def artist_signature_score(track: Track, profile: ArtistProfile) -> float:
    """Score how completely a track covers its artist's stable signature."""

    weights = profile.weights
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    track_tags = set(track.style_tags)
    return sum(weight for tag, weight in weights.items() if tag in track_tags) / total


def choose_artists_balanced(
    artist_tracks: Mapping[str, Sequence[Track]],
    profiles: Mapping[str, ArtistProfile],
    count: int,
    *,
    seed: int,
) -> list[str]:
    """Choose artists round-robin across the 15 dominant coarse genres."""

    if count < 1:
        raise ValueError("artist count must be positive")
    if len(artist_tracks) < count:
        raise ValueError(
            f"requested {count} artists, but only {len(artist_tracks)} qualify"
        )

    buckets: dict[str, list[str]] = defaultdict(list)
    for artist_key in artist_tracks:
        buckets[profiles[artist_key].dominant_genre].append(artist_key)

    rng = random.Random(seed)
    queues: dict[str, deque[str]] = {}
    for genre, artists in buckets.items():
        rng.shuffle(artists)
        artists.sort(key=lambda key: len(artist_tracks[key]), reverse=True)
        queues[genre] = deque(artists)

    selected: list[str] = []
    bucket_selection_counts: Counter[str] = Counter()
    while len(selected) < count:
        available_genres = [genre for genre, queue in queues.items() if queue]
        if not available_genres:  # Defensive; the size check above should prevent this.
            raise RuntimeError("artist genre buckets were exhausted unexpectedly")
        genre = min(
            available_genres,
            key=lambda name: (
                bucket_selection_counts[name],
                -len(queues[name]),
                COARSE_GENRES.index(name),
            ),
        )
        selected.append(queues[genre].popleft())
        bucket_selection_counts[genre] += 1
    return selected


def scan_captions(
    caption_file: str | Path,
    track_ids: set[str],
    *,
    min_clip_seconds: float = 5.0,
) -> tuple[dict[str, list[Caption]], Counter[str]]:
    """Read the caption file once and retain rows for candidate tracks only."""

    path = Path(caption_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"caption file does not exist: {path}")

    captions: dict[str, list[Caption]] = defaultdict(list)
    seen: dict[str, set[tuple[float, float, str]]] = defaultdict(set)
    stats: Counter[str] = Counter()
    with path.open("rb") as handle, tqdm(
        total=path.stat().st_size,
        desc="Scanning captions",
        unit="B",
        unit_scale=True,
    ) as progress:
        for line in handle:
            progress.update(len(line))
            if not line.strip():
                continue
            stats["caption_rows"] += 1
            try:
                item = _loads(line)
            except (json.JSONDecodeError, ValueError, TypeError):
                stats["caption_parse_errors"] += 1
                continue
            track_id = _clean_text(item.get("id"))
            if track_id not in track_ids:
                continue
            text = _clean_text(item.get("caption"))
            start = _clean_float(item.get("start_time"))
            end = _clean_float(item.get("end_time"))
            if text is None or start is None or end is None:
                stats["invalid_candidate_captions"] += 1
                continue
            if start < 0 or end - start < min_clip_seconds:
                stats["invalid_candidate_captions"] += 1
                continue
            key = (start, end, text)
            if key in seen[track_id]:
                stats["duplicate_candidate_captions"] += 1
                continue
            seen[track_id].add(key)
            captions[track_id].append(Caption(start, end, text))
            stats["matched_caption_rows"] += 1

    for track_captions in captions.values():
        track_captions.sort(key=lambda caption: (caption.start_time, caption.end_time))
    stats["tracks_with_matched_captions"] = len(captions)
    return dict(captions), stats


def usable_captions(
    track: Track,
    captions: Sequence[Caption],
    *,
    min_clip_seconds: float,
    max_clips_per_song: int | None,
) -> tuple[Caption, ...]:
    """Clamp caption spans to metadata duration and drop unusably short clips."""

    result: list[Caption] = []
    for caption in captions:
        end = caption.end_time
        if track.duration is not None and track.duration > 0:
            end = min(end, track.duration)
        if end - caption.start_time < min_clip_seconds:
            continue
        result.append(Caption(caption.start_time, end, caption.text))
    result.sort(key=lambda value: (value.start_time, value.end_time))
    if max_clips_per_song is not None:
        result = result[:max_clips_per_song]
    return tuple(result)


def select_representative_tracks(
    artist_keys: Sequence[str],
    artist_tracks: Mapping[str, Sequence[Track]],
    profiles: Mapping[str, ArtistProfile],
    captions_by_track: Mapping[str, Sequence[Caption]],
    songs_per_artist: int,
    *,
    seed: int,
    min_clip_seconds: float,
    max_clips_per_song: int | None,
    max_songs_per_album: int,
) -> list[SelectedSong]:
    """Select each artist's highest-signature songs with album diversity."""

    if max_songs_per_album < 1:
        raise ValueError("max_songs_per_album must be positive")
    rng = random.Random(seed)
    selected: list[SelectedSong] = []
    for artist_key in artist_keys:
        profile = profiles[artist_key]
        candidates: list[tuple[Track, tuple[Caption, ...], float]] = []
        for track in artist_tracks[artist_key]:
            captions = usable_captions(
                track,
                captions_by_track.get(track.track_id, ()),
                min_clip_seconds=min_clip_seconds,
                max_clips_per_song=max_clips_per_song,
            )
            if captions:
                candidates.append(
                    (track, captions, artist_signature_score(track, profile))
                )
        if len(candidates) < songs_per_artist:
            raise ValueError(
                f"artist {artist_key} has only {len(candidates)} captioned tracks; "
                f"{songs_per_artist} required"
            )
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda item: (
                -item[2],
                -len(set(item[0].style_tags) & set(profile.signature_tags)),
                item[0].track_id,
            )
        )

        chosen: list[tuple[Track, tuple[Caption, ...], float]] = []
        album_counts: Counter[str] = Counter()
        for item in candidates:
            album_id = item[0].album_id
            if album_id is not None and album_counts[album_id] >= max_songs_per_album:
                continue
            chosen.append(item)
            if album_id is not None:
                album_counts[album_id] += 1
            if len(chosen) == songs_per_artist:
                break
        # A small catalog may require relaxing the album cap.  Signature score
        # ordering is preserved, so the fallback remains representative.
        if len(chosen) < songs_per_artist:
            chosen_ids = {item[0].track_id for item in chosen}
            for item in candidates:
                if item[0].track_id in chosen_ids:
                    continue
                chosen.append(item)
                if len(chosen) == songs_per_artist:
                    break

        for track, captions, score in chosen:
            selected.append(
                SelectedSong(
                    track=track,
                    balance_genre=profile.dominant_genre,
                    captions=captions,
                    signature_score=score,
                    signature_tags=profile.signature_tags,
                )
            )
    return selected


def build_selection(
    metadata_files: Sequence[str | Path],
    caption_file: str | Path,
    *,
    num_artists: int,
    songs_per_artist: int,
    seed: int = 42,
    artist_candidate_multiplier: float = 3.0,
    min_clip_seconds: float = 5.0,
    max_clips_per_song: int | None = None,
    signature_min_support: float = 0.30,
    signature_min_songs: int = 3,
    max_signature_tags: int = 5,
    min_global_style_tag_count: int = 50,
    signature_idf_cap: float = 3.0,
    max_songs_per_album: int = 2,
) -> tuple[list[SelectedSong], dict[str, Any]]:
    """Balance coarse artist genres and select signature-representative songs."""

    if num_artists < 1 or songs_per_artist < 1:
        raise ValueError("num_artists and songs_per_artist must both be positive")
    if artist_candidate_multiplier < 1:
        raise ValueError("artist_candidate_multiplier must be at least 1")
    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be positive")
    if max_clips_per_song is not None and max_clips_per_song < 1:
        raise ValueError("max_clips_per_song must be positive when supplied")

    all_artist_tracks, metadata_stats = scan_metadata(metadata_files)
    raw_to_coarse, taxonomy_stats = build_coarse_genre_taxonomy(all_artist_tracks)
    all_artist_tracks, dropped_unresolved_tracks = apply_coarse_genre_taxonomy(
        all_artist_tracks, raw_to_coarse
    )
    profiles = build_artist_profiles(
        all_artist_tracks,
        signature_min_support=signature_min_support,
        signature_min_songs=signature_min_songs,
        max_signature_tags=max_signature_tags,
        min_global_style_tag_count=min_global_style_tag_count,
        signature_idf_cap=signature_idf_cap,
    )
    # Artist balancing is meaningful only when the selected songs themselves
    # belong to the artist's dominant coarse family and cover at least one
    # stable signature tag.  Profiles still come from the full catalog.
    eligible_artist_tracks = {}
    for key, tracks in all_artist_tracks.items():
        profile = profiles[key]
        representative_tracks = [
            track
            for track in tracks
            if profile.dominant_genre in track.coarse_genres
            and artist_signature_score(track, profile) > 0
        ]
        if len(representative_tracks) >= songs_per_artist:
            eligible_artist_tracks[key] = representative_tracks
    if len(eligible_artist_tracks) < num_artists:
        raise ValueError(
            f"only {len(eligible_artist_tracks)} artists have at least "
            f"{songs_per_artist} downloadable, genre-tagged songs; "
            f"{num_artists} requested"
        )

    candidate_count = min(
        len(eligible_artist_tracks),
        max(
            num_artists,
            math.ceil(num_artists * artist_candidate_multiplier),
            num_artists + 20,
        ),
    )
    candidate_artist_keys = choose_artists_balanced(
        eligible_artist_tracks,
        profiles,
        candidate_count,
        seed=seed,
    )
    candidate_tracks = {
        key: eligible_artist_tracks[key] for key in candidate_artist_keys
    }
    candidate_track_ids = {
        track.track_id for tracks in candidate_tracks.values() for track in tracks
    }
    captions_by_track, caption_stats = scan_captions(
        caption_file,
        candidate_track_ids,
        min_clip_seconds=min_clip_seconds,
    )

    caption_qualified_tracks: dict[str, list[Track]] = {}
    for artist_key, tracks in candidate_tracks.items():
        usable_tracks = [
            track
            for track in tracks
            if usable_captions(
                track,
                captions_by_track.get(track.track_id, ()),
                min_clip_seconds=min_clip_seconds,
                max_clips_per_song=max_clips_per_song,
            )
        ]
        if len(usable_tracks) >= songs_per_artist:
            caption_qualified_tracks[artist_key] = usable_tracks

    if len(caption_qualified_tracks) < num_artists:
        raise ValueError(
            f"only {len(caption_qualified_tracks)} of {candidate_count} candidate "
            f"artists have {songs_per_artist} captioned songs. Increase "
            "--artist-candidate-multiplier and retry."
        )

    final_artist_keys = choose_artists_balanced(
        caption_qualified_tracks,
        profiles,
        num_artists,
        seed=seed + 1,
    )
    selected = select_representative_tracks(
        final_artist_keys,
        caption_qualified_tracks,
        profiles,
        captions_by_track,
        songs_per_artist,
        seed=seed + 2,
        min_clip_seconds=min_clip_seconds,
        max_clips_per_song=max_clips_per_song,
        max_songs_per_album=max_songs_per_album,
    )

    artist_coarse_counts = Counter(
        profiles[artist_key].dominant_genre for artist_key in final_artist_keys
    )
    song_coarse_counts = Counter(song.balance_genre for song in selected)
    signature_scores = [song.signature_score for song in selected]
    signature_type_counts = Counter(
        tag.split(":", 1)[0]
        for artist_key in final_artist_keys
        for tag in profiles[artist_key].signature_tags
    )
    selection_stats: dict[str, Any] = {
        "metadata": dict(metadata_stats),
        "taxonomy": {
            **taxonomy_stats,
            "tracks_dropped_without_coarse_genre": dropped_unresolved_tracks,
        },
        "captions": dict(caption_stats),
        "artists_meeting_metadata_requirement": len(eligible_artist_tracks),
        "candidate_artists_checked_for_captions": candidate_count,
        "caption_qualified_candidate_artists": len(caption_qualified_tracks),
        "selected_artists": len({song.track.artist_key for song in selected}),
        "selected_songs": len(selected),
        "selected_clips": sum(len(song.captions) for song in selected),
        "coarse_genre_artist_counts": dict(sorted(artist_coarse_counts.items())),
        "coarse_genre_song_counts": dict(sorted(song_coarse_counts.items())),
        "balance_genre_counts": dict(sorted(song_coarse_counts.items())),
        "signature_score": {
            "min": min(signature_scores),
            "mean": sum(signature_scores) / len(signature_scores),
            "median": statistics.median(signature_scores),
            "max": max(signature_scores),
        },
        "signature_tag_type_counts": dict(sorted(signature_type_counts.items())),
        "selected_artist_profiles": {
            artist_key: {
                "dominant_genre": profiles[artist_key].dominant_genre,
                "signature_tags": list(profiles[artist_key].signature_tags),
                "signature_supports": dict(profiles[artist_key].signature_supports),
                "signature_weights": dict(profiles[artist_key].signature_weights),
            }
            for artist_key in final_artist_keys
        },
    }
    return selected, selection_stats


def default_subset_name(num_artists: int, songs_per_artist: int, seed: int) -> str:
    return (
        f"signature_artists{num_artists:04d}_songs{songs_per_artist:03d}"
        f"_seed{seed:04d}"
    )


def _track_record(song: SelectedSong) -> dict[str, Any]:
    track = song.track
    return {
        "track_id": track.track_id,
        "title": track.title,
        "artist_key": track.artist_key,
        "artist_id": track.artist_id,
        "artist_name": track.artist_name,
        "album_id": track.album_id,
        "album_name": track.album_name,
        "duration": track.duration,
        "releasedate": track.releasedate,
        "audiodownload": track.audiodownload,
        "genres": list(track.genres),
        "coarse_genres": list(track.coarse_genres),
        "style_tags": list(track.style_tags),
        "coarse_genre": song.balance_genre,
        "signature_score": song.signature_score,
        "signature_tags": list(song.signature_tags),
        "captioned_clips": len(song.captions),
    }


def _caption_record(song: SelectedSong, caption: Caption) -> dict[str, Any]:
    return {
        "track_id": song.track.track_id,
        "artist_id": song.track.artist_id,
        "artist_name": song.track.artist_name,
        "start_time": caption.start_time,
        "end_time": caption.end_time,
        "duration": caption.duration,
        "caption": caption.text,
        "genres": list(song.track.genres),
        "coarse_genres": list(song.track.coarse_genres),
        "coarse_genre": song.balance_genre,
        "signature_score": song.signature_score,
        "signature_tags": list(song.signature_tags),
    }


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(target)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(target)


def write_selection_metadata(
    output_dir: str | Path,
    selected: Sequence[SelectedSong],
    selection_stats: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    root = Path(output_dir)
    metadata_dir = root / "metadata"
    write_json(metadata_dir / "config.json", dict(config))
    write_json(metadata_dir / "selection_stats.json", dict(selection_stats))
    write_jsonl(metadata_dir / "songs.jsonl", (_track_record(song) for song in selected))
    write_jsonl(
        metadata_dir / "captions.jsonl",
        (
            _caption_record(song, caption)
            for song in selected
            for caption in song.captions
        ),
    )

    songs_by_artist: dict[str, list[SelectedSong]] = defaultdict(list)
    for song in selected:
        songs_by_artist[song.track.artist_key].append(song)
    artist_rows = []
    for artist_key, songs in sorted(songs_by_artist.items()):
        first = songs[0].track
        artist_rows.append(
            {
                "artist_key": artist_key,
                "artist_id": first.artist_id,
                "artist_name": first.artist_name,
                "songs": len(songs),
                "clips": sum(len(song.captions) for song in songs),
                "track_ids": [song.track.track_id for song in songs],
                "genres": sorted({genre for song in songs for genre in song.track.genres}),
                "coarse_genre": songs[0].balance_genre,
                "signature_tags": list(songs[0].signature_tags),
                "signature_score_mean": sum(song.signature_score for song in songs)
                / len(songs),
            }
        )
    write_jsonl(metadata_dir / "artists.jsonl", artist_rows)


_thread_local = threading.local()


def _http_session(retries: int, pool_size: int) -> requests.Session:
    key = (retries, pool_size)
    if getattr(_thread_local, "session_key", None) == key:
        return _thread_local.session
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _thread_local.session_key = key
    _thread_local.session = session
    return session


def source_audio_path(output_dir: str | Path, track: Track) -> Path:
    return (
        Path(output_dir)
        / "source_audio"
        / artist_directory(track)
        / f"track_{safe_path_component(track.track_id, fallback_prefix='track')}.mp3"
    )


def looks_like_mp3_header(prefix: bytes) -> bool:
    """Recognize an ID3 tag or MPEG audio frame sync at the start of a file."""

    return prefix.startswith(b"ID3") or (
        len(prefix) >= 2 and prefix[0] == 0xFF and prefix[1] & 0xE0 == 0xE0
    )


def download_track(
    song: SelectedSong,
    output_dir: str | Path,
    *,
    retries: int,
    timeout_seconds: float,
    pool_size: int,
) -> DownloadResult:
    """Download one track atomically, or reuse a valid existing file."""

    destination = source_audio_path(output_dir, song.track)
    if destination.is_file() and destination.stat().st_size >= MIN_AUDIO_BYTES:
        return DownloadResult(song.track.track_id, destination, "cached")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        if temporary.exists():
            temporary.unlink()
        session = _http_session(retries, pool_size)
        with session.get(
            song.track.audiodownload,
            stream=True,
            timeout=(10.0, timeout_seconds),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").casefold()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if temporary.stat().st_size < MIN_AUDIO_BYTES:
            raise RuntimeError(
                f"downloaded file is only {temporary.stat().st_size} bytes"
            )
        with temporary.open("rb") as handle:
            prefix = handle.read(16)
        # Jamendo currently serves valid MP3 bytes with a misleading text/html
        # Content-Type.  Validate the payload magic instead of trusting that header.
        if not looks_like_mp3_header(prefix):
            raise RuntimeError(
                f"response is not an MP3 (Content-Type: {content_type or 'missing'})"
            )
        temporary.replace(destination)
        return DownloadResult(song.track.track_id, destination, "downloaded")
    except Exception as error:  # noqa: BLE001 - preserve per-track failures for resume
        if temporary.exists():
            temporary.unlink()
        return DownloadResult(
            song.track.track_id,
            destination,
            "failed",
            f"{type(error).__name__}: {error}",
        )


def download_selected_audio(
    selected: Sequence[SelectedSong],
    output_dir: str | Path,
    *,
    workers: int = 16,
    retries: int = 4,
    timeout_seconds: float = 120.0,
) -> dict[str, DownloadResult]:
    """Download selected songs concurrently and return results by track id.

    A track whose complete set of caption clips is already present does not need
    its deleted source MP3 to be downloaded again.  Treat it as successful so a
    resumed run can rebuild the manifest from the cached clips.
    """

    if workers < 1:
        raise ValueError("download workers must be positive")
    results: dict[str, DownloadResult] = {}
    pending: list[SelectedSong] = []
    for song in selected:
        clips_complete = bool(song.captions) and all(
            (
                (path := clip_output_path(output_dir, song, caption)).is_file()
                and path.stat().st_size >= MIN_CLIP_BYTES
            )
            for caption in song.captions
        )
        if clips_complete:
            source = source_audio_path(output_dir, song.track)
            stale_part = source.with_suffix(source.suffix + ".part")
            try:
                stale_part.unlink()
            except FileNotFoundError:
                pass
            except PermissionError:
                # Another concurrent/residual downloader may still own the
                # temporary file.  Completed clips remain sufficient to resume.
                pass
            results[song.track.track_id] = DownloadResult(
                song.track.track_id,
                source,
                "clips_cached",
            )
        else:
            pending.append(song)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_track,
                song,
                output_dir,
                retries=retries,
                timeout_seconds=timeout_seconds,
                pool_size=workers,
            ): song.track.track_id
            for song in pending
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading songs",
            unit="song",
        ):
            result = future.result()
            results[result.track_id] = result
    return results


def clip_output_path(output_dir: str | Path, song: SelectedSong, caption: Caption) -> Path:
    start_ms = round(caption.start_time * 1000)
    end_ms = round(caption.end_time * 1000)
    filename = f"{safe_path_component(song.track.track_id)}_{start_ms:09d}_{end_ms:09d}.flac"
    return (
        Path(output_dir)
        / "clips"
        / artist_directory(song.track)
        / f"track_{safe_path_component(song.track.track_id)}"
        / filename
    )


def build_clip_tasks(
    selected: Sequence[SelectedSong],
    downloads: Mapping[str, DownloadResult],
    output_dir: str | Path,
) -> list[ClipTask]:
    tasks: list[ClipTask] = []
    for song in selected:
        download = downloads.get(song.track.track_id)
        if download is None or not download.ok:
            continue
        for caption in song.captions:
            tasks.append(
                ClipTask(
                    song=song,
                    caption=caption,
                    source_path=download.path,
                    output_path=clip_output_path(output_dir, song, caption),
                )
            )
    return tasks


class _FfmpegProcessRegistry:
    """Track active FFmpeg children so Ctrl+C can stop them promptly."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[Any]] = set()
        self._stopping = False

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping

    def register(self, process: subprocess.Popen[Any]) -> bool:
        with self._lock:
            if self._stopping:
                should_stop = True
            else:
                self._processes.add(process)
                should_stop = False
        if should_stop:
            try:
                process.terminate()
            except OSError:
                pass
            return False
        return True

    def unregister(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self, grace_seconds: float = 2.0) -> None:
        """Prevent new children, terminate active ones, then kill stragglers."""

        with self._lock:
            self._stopping = True
            processes = tuple(self._processes)
        for process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
            except OSError:
                pass

        deadline = time.monotonic() + grace_seconds
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass


def clip_one(
    task: ClipTask,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    timeout_seconds: float = 300.0,
    process_registry: _FfmpegProcessRegistry | None = None,
) -> ClipResult:
    """Decode exactly one caption span to a lossless MusicGen-ready FLAC."""

    destination = task.output_path
    if destination.is_file() and destination.stat().st_size >= MIN_CLIP_BYTES:
        return ClipResult(task, "cached")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{task.caption.start_time:.6f}",
        "-t",
        f"{task.caption.duration:.6f}",
        "-i",
        str(task.source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "flac",
        "-sample_fmt",
        DEFAULT_SAMPLE_FORMAT,
        "-compression_level",
        "5",
        "-threads",
        "1",
        str(temporary),
    ]
    try:
        if temporary.exists():
            temporary.unlink()
        if process_registry is not None and process_registry.stopping:
            raise InterruptedError("FFmpeg clipping was interrupted")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        registered = process_registry is None or process_registry.register(process)
        try:
            if not registered:
                process.communicate()
                raise InterruptedError("FFmpeg clipping was interrupted")
            try:
                _, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                process.kill()
                _, stderr = process.communicate()
                detail = (stderr or "").strip()[-1000:]
                raise RuntimeError(
                    f"ffmpeg timed out after {timeout_seconds:g}s: {detail}"
                ) from error
        finally:
            if process_registry is not None:
                process_registry.unregister(process)
        if process.returncode != 0:
            detail = (stderr or "").strip()[-1000:]
            raise RuntimeError(f"ffmpeg exited with {process.returncode}: {detail}")
        if not temporary.is_file() or temporary.stat().st_size < MIN_CLIP_BYTES:
            raise RuntimeError("ffmpeg did not produce a valid-sized clip")
        temporary.replace(destination)
        return ClipResult(task, "created")
    except Exception as error:  # noqa: BLE001 - preserve failures in a manifest
        if temporary.exists():
            temporary.unlink()
        return ClipResult(task, "failed", f"{type(error).__name__}: {error}")


def create_clips(
    tasks: Sequence[ClipTask],
    *,
    workers: int = 4,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    timeout_seconds: float = 300.0,
    max_in_flight: int | None = None,
) -> list[ClipResult]:
    """Create clips with a bounded queue and promptly interruptible FFmpeg.

    Only a small multiple of ``workers`` is submitted at once.  This avoids
    allocating tens of thousands of Future objects before tqdm can render and
    lets Ctrl+C cancel the short pending queue rather than waiting for every
    clip in the dataset.
    """

    if workers < 1:
        raise ValueError("ffmpeg workers must be positive")
    if max_in_flight is None:
        max_in_flight = workers * 2
    if max_in_flight < 1:
        raise ValueError("max in-flight clips must be positive")

    results: list[ClipResult] = []
    registry = _FfmpegProcessRegistry()
    executor = ThreadPoolExecutor(max_workers=workers)
    task_iterator = iter(tasks)
    pending: dict[Any, ClipTask] = {}

    def submit_next() -> bool:
        try:
            task = next(task_iterator)
        except StopIteration:
            return False
        future = executor.submit(
            clip_one,
            task,
            ffmpeg=ffmpeg,
            sample_rate=sample_rate,
            channels=channels,
            timeout_seconds=timeout_seconds,
            process_registry=registry,
        )
        pending[future] = task
        return True

    try:
        with tqdm(
            total=len(tasks),
            desc="Creating caption clips",
            unit="clip",
        ) as progress:
            for _ in range(min(max_in_flight, len(tasks))):
                submit_next()
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    results.append(future.result())
                    progress.update(1)
                    submit_next()
    except BaseException:
        registry.terminate_all()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def _manifest_record(
    result: ClipResult,
    output_dir: Path,
    *,
    sample_rate: int,
    channels: int,
) -> dict[str, Any]:
    task = result.task
    song = task.song
    return {
        "audio": task.output_path.relative_to(output_dir).as_posix(),
        "text": task.caption.text,
        "track_id": song.track.track_id,
        "artist_key": song.track.artist_key,
        "artist_id": song.track.artist_id,
        "artist_name": song.track.artist_name,
        "title": song.track.title,
        "genres": list(song.track.genres),
        "coarse_genres": list(song.track.coarse_genres),
        "coarse_genre": song.balance_genre,
        "signature_score": song.signature_score,
        "signature_tags": list(song.signature_tags),
        "start_time": task.caption.start_time,
        "end_time": task.caption.end_time,
        "duration": task.caption.duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_format": DEFAULT_SAMPLE_FORMAT,
        "bits_per_sample": DEFAULT_BITS_PER_SAMPLE,
    }


def finalize_outputs(
    output_dir: str | Path,
    selected: Sequence[SelectedSong],
    downloads: Mapping[str, DownloadResult],
    clip_results: Sequence[ClipResult],
    *,
    sample_rate: int,
    channels: int,
    keep_originals: bool,
) -> dict[str, Any]:
    """Write manifests/failures, optionally remove sources, and return run stats."""

    root = Path(output_dir).resolve()
    metadata_dir = root / "metadata"
    successful_clips = sorted(
        (result for result in clip_results if result.ok),
        key=lambda result: (
            result.task.song.track.artist_key,
            result.task.song.track.track_id,
            result.task.caption.start_time,
        ),
    )
    write_jsonl(
        metadata_dir / "manifest.jsonl",
        (
            _manifest_record(
                result,
                root,
                sample_rate=sample_rate,
                channels=channels,
            )
            for result in successful_clips
        ),
    )

    download_failures = [
        {
            "track_id": result.track_id,
            "url": next(
                song.track.audiodownload
                for song in selected
                if song.track.track_id == result.track_id
            ),
            "error": result.error,
        }
        for result in downloads.values()
        if not result.ok
    ]
    clip_failures = [
        {
            "track_id": result.task.song.track.track_id,
            "start_time": result.task.caption.start_time,
            "end_time": result.task.caption.end_time,
            "output": str(result.task.output_path),
            "error": result.error,
        }
        for result in clip_results
        if not result.ok
    ]
    write_jsonl(metadata_dir / "download_failures.jsonl", download_failures)
    write_jsonl(metadata_dir / "clip_failures.jsonl", clip_failures)

    if not keep_originals:
        failed_track_ids = {row["track_id"] for row in clip_failures}
        expected_by_track = Counter(
            song.track.track_id for song in selected for _ in song.captions
        )
        completed_by_track = Counter(
            result.task.song.track.track_id for result in successful_clips
        )
        for track_id, download in downloads.items():
            if (
                download.ok
                and track_id not in failed_track_ids
                and completed_by_track[track_id] == expected_by_track[track_id]
                and download.path.exists()
            ):
                download.path.unlink()

    stats = {
        "selected_artists": len({song.track.artist_key for song in selected}),
        "selected_songs": len(selected),
        "expected_clips": sum(len(song.captions) for song in selected),
        "successful_downloads": sum(result.ok for result in downloads.values()),
        "failed_downloads": len(download_failures),
        "successful_clips": len(successful_clips),
        "failed_clips": len(clip_failures),
        "kept_source_audio": keep_originals,
        "sample_format": DEFAULT_SAMPLE_FORMAT,
        "bits_per_sample": DEFAULT_BITS_PER_SAMPLE,
        "manifest": "metadata/manifest.jsonl",
    }
    write_json(metadata_dir / "run_stats.json", stats)
    return stats


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset_dir = repo_root / "datasets" / "jamendo_max_caps"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-artists", type=int, required=True)
    parser.add_argument("--songs-per-artist", type=int, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="JamendoMaxCaps root containing the metadata directory",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Override the date-partitioned metadata directory",
    )
    parser.add_argument(
        "--caption-file",
        type=Path,
        default=None,
        help="Override final_caption30sec.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Subset output; default name encodes artists, songs, and seed",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--artist-candidate-multiplier",
        type=float,
        default=3.0,
        help="Artists checked for captions relative to the requested count",
    )
    parser.add_argument("--min-clip-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-clips-per-song",
        type=int,
        default=None,
        help="Optional cap on chronological caption clips retained per song",
    )
    parser.add_argument(
        "--signature-min-support",
        type=float,
        default=0.30,
        help="Minimum fraction of an artist catalog containing a signature tag",
    )
    parser.add_argument(
        "--signature-min-songs",
        type=int,
        default=3,
        help="Minimum artist songs containing a stable signature tag",
    )
    parser.add_argument("--max-signature-tags", type=int, default=5)
    parser.add_argument(
        "--min-global-style-tag-count",
        type=int,
        default=50,
        help="Ignore globally ultra-rare style tags unless local fallback is needed",
    )
    parser.add_argument("--signature-idf-cap", type=float, default=3.0)
    parser.add_argument(
        "--max-songs-per-album",
        type=int,
        default=2,
        help="Album diversity cap, relaxed only when the catalog is too small",
    )
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--ffmpeg-workers", type=int, default=4)
    parser.add_argument(
        "--max-in-flight-clips",
        type=int,
        default=None,
        help="Maximum running/queued clip tasks; defaults to 2x ffmpeg workers",
    )
    parser.add_argument("--download-retries", type=int, default=4)
    parser.add_argument("--download-timeout", type=float, default=120.0)
    parser.add_argument("--ffmpeg-timeout", type=float, default=300.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep downloaded source MP3s after all clips for a song succeed",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write selection/caption metadata but do not download or clip audio",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_in_flight_clips = (
        args.max_in_flight_clips
        if args.max_in_flight_clips is not None
        else args.ffmpeg_workers * 2
    )
    dataset_dir = args.dataset_dir.expanduser().resolve()
    metadata_dir = (
        args.metadata_dir.expanduser().resolve()
        if args.metadata_dir is not None
        else dataset_dir / "metadata"
    )
    caption_file = (
        args.caption_file.expanduser().resolve()
        if args.caption_file is not None
        else metadata_dir / "final_caption30sec.jsonl"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_dir
        / "subsets"
        / default_subset_name(args.num_artists, args.songs_per_artist, args.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which(args.ffmpeg) is None and not args.selection_only:
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg}")

    metadata_files = discover_metadata_files(metadata_dir)
    config = {
        "num_artists": args.num_artists,
        "songs_per_artist": args.songs_per_artist,
        "seed": args.seed,
        "dataset_dir": str(dataset_dir),
        "metadata_dir": str(metadata_dir),
        "caption_file": str(caption_file),
        "output_dir": str(output_dir),
        "artist_candidate_multiplier": args.artist_candidate_multiplier,
        "min_clip_seconds": args.min_clip_seconds,
        "max_clips_per_song": args.max_clips_per_song,
        "signature_min_support": args.signature_min_support,
        "signature_min_songs": args.signature_min_songs,
        "max_signature_tags": args.max_signature_tags,
        "min_global_style_tag_count": args.min_global_style_tag_count,
        "signature_idf_cap": args.signature_idf_cap,
        "max_songs_per_album": args.max_songs_per_album,
        "download_workers": args.download_workers,
        "ffmpeg_workers": args.ffmpeg_workers,
        "max_in_flight_clips": max_in_flight_clips,
        "sample_rate": args.sample_rate,
        "channels": args.channels,
        "sample_format": DEFAULT_SAMPLE_FORMAT,
        "bits_per_sample": DEFAULT_BITS_PER_SAMPLE,
        "keep_originals": args.keep_originals,
        "selection_only": args.selection_only,
    }

    print(f"Metadata files: {len(metadata_files):,}")
    print(f"Caption file: {caption_file}")
    print(f"Output: {output_dir}")
    selected, selection_stats = build_selection(
        metadata_files,
        caption_file,
        num_artists=args.num_artists,
        songs_per_artist=args.songs_per_artist,
        seed=args.seed,
        artist_candidate_multiplier=args.artist_candidate_multiplier,
        min_clip_seconds=args.min_clip_seconds,
        max_clips_per_song=args.max_clips_per_song,
        signature_min_support=args.signature_min_support,
        signature_min_songs=args.signature_min_songs,
        max_signature_tags=args.max_signature_tags,
        min_global_style_tag_count=args.min_global_style_tag_count,
        signature_idf_cap=args.signature_idf_cap,
        max_songs_per_album=args.max_songs_per_album,
    )
    write_selection_metadata(output_dir, selected, selection_stats, config)
    print(
        f"Selected {selection_stats['selected_artists']:,} artists, "
        f"{selection_stats['selected_songs']:,} songs, and "
        f"{selection_stats['selected_clips']:,} caption clips."
    )

    if args.selection_only:
        print("Selection-only mode: audio download and clipping skipped.")
        return 0

    downloads = download_selected_audio(
        selected,
        output_dir,
        workers=args.download_workers,
        retries=args.download_retries,
        timeout_seconds=args.download_timeout,
    )
    tasks = build_clip_tasks(selected, downloads, output_dir)
    try:
        clip_results = create_clips(
            tasks,
            workers=args.ffmpeg_workers,
            ffmpeg=args.ffmpeg,
            sample_rate=args.sample_rate,
            channels=args.channels,
            timeout_seconds=args.ffmpeg_timeout,
            max_in_flight=max_in_flight_clips,
        )
    except KeyboardInterrupt:
        print("\nInterrupted: active FFmpeg processes stopped; rerun to resume.")
        return 130
    run_stats = finalize_outputs(
        output_dir,
        selected,
        downloads,
        clip_results,
        sample_rate=args.sample_rate,
        channels=args.channels,
        keep_originals=args.keep_originals,
    )
    print(json.dumps(run_stats, ensure_ascii=False, indent=2))
    if run_stats["failed_downloads"] or run_stats["failed_clips"]:
        print("Some items failed; rerun the same command to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
