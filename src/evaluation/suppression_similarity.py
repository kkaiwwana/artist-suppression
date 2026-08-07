"""Embedding-based diagnostics for explicit artist suppression.

The helpers in this module deliberately avoid artist classifiers.  CLAP
measures whether text alignment survives suppression, while a music audio
encoder such as MERT measures paired output drift and movement relative to
real audio from each evaluation artist.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _artist_key(record: Mapping[str, Any]) -> str:
    for field in ("artist_key", "artist_id", "artist_name"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("record has no artist identifier")


def _genres(record: Mapping[str, Any], genre_field: str) -> list[str]:
    value = record.get(genre_field)
    if value is None and genre_field == "coarse_genres":
        value = record.get("coarse_genre")
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    return sorted({str(item).strip() for item in values if str(item).strip()})


def build_artist_profiles(
    records: Sequence[Mapping[str, Any]],
    artist_to_index: Mapping[str, int],
    *,
    genre_field: str = "coarse_genres",
) -> list[dict[str, Any]]:
    """Summarize available clips and dominant genre for every artist."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_artist_key(record)].append(record)

    profiles: list[dict[str, Any]] = []
    for key, artist_records in grouped.items():
        if key not in artist_to_index:
            continue
        genre_counts = Counter(
            genre
            for record in artist_records
            for genre in _genres(record, genre_field)
        )
        dominant_genre = (
            sorted(genre_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if genre_counts
            else "unknown"
        )
        names = Counter(
            str(record.get("artist_name", "")).strip()
            for record in artist_records
            if str(record.get("artist_name", "")).strip()
        )
        profiles.append(
            {
                "artist_key": key,
                "artist_index": int(artist_to_index[key]),
                "artist_name": names.most_common(1)[0][0] if names else key,
                "dominant_genre": dominant_genre,
                "genres": sorted(genre_counts),
                "num_clips": len(artist_records),
            }
        )
    return sorted(profiles, key=lambda row: row["artist_index"])


def _resolve_artist(
    value: str | int,
    profiles: Sequence[Mapping[str, Any]],
) -> str:
    by_key = {str(row["artist_key"]): str(row["artist_key"]) for row in profiles}
    by_index = {int(row["artist_index"]): str(row["artist_key"]) for row in profiles}
    by_name: dict[str, list[str]] = defaultdict(list)
    for row in profiles:
        by_name[str(row["artist_name"]).casefold()].append(str(row["artist_key"]))
    if isinstance(value, int):
        if value not in by_index:
            raise KeyError(f"unknown artist index: {value}")
        return by_index[value]
    text = str(value).strip()
    if text in by_key:
        return text
    matches = by_name.get(text.casefold(), [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"artist name is ambiguous: {value!r}; use artist_key")
    raise KeyError(f"unknown artist key/name: {value!r}")


def choose_evaluation_artists(
    profiles: Sequence[Mapping[str, Any]],
    *,
    clips_per_artist: int,
    seed: int = 42,
    target_artists: Sequence[str | int] | None = None,
    control_artists: Sequence[str | int] | None = None,
    num_targets: int = 2,
    num_controls: int = 2,
) -> tuple[list[str], list[str]]:
    """Choose two genre-distinct targets and unrelated control artists.

    Explicit values may be artist keys, unique artist names, or integer label
    indices.  Automatic target selection prefers artists with many clips and
    enforces a different dominant genre for the second target when possible.
    """

    if clips_per_artist <= 0:
        raise ValueError("clips_per_artist must be positive")
    eligible = [row for row in profiles if int(row["num_clips"]) >= clips_per_artist]
    if len(eligible) < num_targets + num_controls:
        raise ValueError(
            f"need {num_targets + num_controls} artists with at least "
            f"{clips_per_artist} clips; found {len(eligible)}"
        )
    eligible_keys = {str(row["artist_key"]) for row in eligible}
    profile_by_key = {str(row["artist_key"]): row for row in eligible}

    if target_artists:
        targets = [_resolve_artist(value, profiles) for value in target_artists]
        if len(targets) != num_targets:
            raise ValueError(f"target_artists must contain {num_targets} artists")
    else:
        ranked = sorted(
            eligible,
            key=lambda row: (-int(row["num_clips"]), int(row["artist_index"])),
        )
        targets = [str(ranked[0]["artist_key"])]
        while len(targets) < num_targets:
            used_genres = {
                str(profile_by_key[key]["dominant_genre"]) for key in targets
            }
            candidate = next(
                (
                    row
                    for row in ranked
                    if str(row["artist_key"]) not in targets
                    and str(row["dominant_genre"]) not in used_genres
                ),
                next(
                    row for row in ranked if str(row["artist_key"]) not in targets
                ),
            )
            targets.append(str(candidate["artist_key"]))

    if len(set(targets)) != len(targets) or any(key not in eligible_keys for key in targets):
        raise ValueError("target artists must be distinct and have enough clips")

    if control_artists:
        controls = [_resolve_artist(value, profiles) for value in control_artists]
        if len(controls) != num_controls:
            raise ValueError(f"control_artists must contain {num_controls} artists")
    else:
        candidates = sorted(eligible_keys.difference(targets))
        random.Random(seed).shuffle(candidates)
        controls = candidates[:num_controls]
    if (
        len(set(controls)) != len(controls)
        or set(controls).intersection(targets)
        or any(key not in eligible_keys for key in controls)
    ):
        raise ValueError("control artists must be distinct, eligible, and not targets")
    return targets, controls


def sample_artist_indices(
    records: Sequence[Mapping[str, Any]],
    artist_key: str,
    *,
    count: int,
    seed: int,
) -> list[int]:
    """Deterministically sample record indices for one artist."""

    indices = [index for index, record in enumerate(records) if _artist_key(record) == artist_key]
    if len(indices) < count:
        raise ValueError(f"artist {artist_key!r} has only {len(indices)} clips, need {count}")
    digest = hashlib.sha256(f"{seed}:{artist_key}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(indices)
    return sorted(indices[:count])


def build_suppression_weights(
    batch_size: int,
    num_concepts: int,
    suppressed_artist_ids: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a normalized multi-hot artist mixture for joint suppression."""

    ids = sorted({int(value) for value in suppressed_artist_ids})
    if batch_size <= 0 or num_concepts <= 0 or not ids:
        raise ValueError("batch_size, num_concepts, and suppressed_artist_ids are required")
    if ids[0] < 0 or ids[-1] >= num_concepts:
        raise IndexError(f"suppressed artist ids must be in [0, {num_concepts})")
    weights = torch.zeros(batch_size, num_concepts, device=device, dtype=dtype)
    weights[:, ids] = 1.0 / len(ids)
    return weights


def cosine_rows(left: Tensor, right: Tensor) -> Tensor:
    """Cosine similarity for paired embedding rows."""

    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError(f"paired embeddings must share [N,D], got {left.shape}, {right.shape}")
    return F.cosine_similarity(left.float(), right.float(), dim=-1)


def compute_similarity_rows(
    *,
    scenario: str,
    cohort_artist_key: str,
    cohort_artist_name: str,
    role: str,
    texts: Sequence[str],
    reference_audio_embeddings: Tensor,
    baseline_audio_embeddings: Tensor,
    suppressed_audio_embeddings: Tensor,
    baseline_clap_scores: Tensor,
    suppressed_clap_scores: Tensor,
) -> list[dict[str, Any]]:
    """Build per-clip CLAP/MERT diagnostics for one artist cohort."""

    count = len(texts)
    tensors = (
        reference_audio_embeddings,
        baseline_audio_embeddings,
        suppressed_audio_embeddings,
    )
    if any(tensor.ndim != 2 or tensor.shape[0] != count for tensor in tensors):
        raise ValueError("all audio embedding tensors must have shape [len(texts), D]")
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("reference/baseline/suppressed embeddings must share shape")
    baseline_clap_scores = torch.as_tensor(baseline_clap_scores).flatten().float()
    suppressed_clap_scores = torch.as_tensor(suppressed_clap_scores).flatten().float()
    if baseline_clap_scores.numel() != count or suppressed_clap_scores.numel() != count:
        raise ValueError("CLAP score vectors must match len(texts)")

    reference = F.normalize(reference_audio_embeddings.float(), dim=-1)
    baseline = F.normalize(baseline_audio_embeddings.float(), dim=-1)
    suppressed = F.normalize(suppressed_audio_embeddings.float(), dim=-1)
    centroid = F.normalize(reference.mean(dim=0, keepdim=True), dim=-1)
    paired = cosine_rows(baseline, suppressed)
    gt_before = cosine_rows(reference, baseline)
    gt_after = cosine_rows(reference, suppressed)
    centroid_before = (baseline * centroid).sum(dim=-1)
    centroid_after = (suppressed * centroid).sum(dim=-1)

    rows: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        rows.append(
            {
                "scenario": scenario,
                "role": role,
                "artist_key": cohort_artist_key,
                "artist_name": cohort_artist_name,
                "clip_index": index,
                "text": str(text),
                "clap_text_before": float(baseline_clap_scores[index]),
                "clap_text_after": float(suppressed_clap_scores[index]),
                "clap_text_delta": float(
                    suppressed_clap_scores[index] - baseline_clap_scores[index]
                ),
                "mert_before_after_cosine": float(paired[index]),
                "mert_gt_before": float(gt_before[index]),
                "mert_gt_after": float(gt_after[index]),
                "mert_gt_delta": float(gt_after[index] - gt_before[index]),
                "mert_artist_centroid_before": float(centroid_before[index]),
                "mert_artist_centroid_after": float(centroid_after[index]),
                "mert_artist_centroid_delta": float(
                    centroid_after[index] - centroid_before[index]
                ),
            }
        )
    return rows


class HFCLAPEncoder:
    """Thin per-sample embedding wrapper around Transformers CLAP."""

    def __init__(
        self,
        model_name_or_path: str = "laion/clap-htsat-unfused",
        *,
        device: torch.device | str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        from transformers import AutoProcessor, ClapModel

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self.model = ClapModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        ).eval().to(self.device)
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.sample_rate = int(getattr(feature_extractor, "sampling_rate", 48_000))

    @staticmethod
    def _mono(audio: Tensor) -> Tensor:
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if audio.ndim != 2:
            raise ValueError("audio must have shape [B,T] or [B,C,T]")
        return audio.detach().float().cpu()

    def _resample(self, audio: Tensor, sample_rate: int) -> Tensor:
        audio = self._mono(audio)
        if int(sample_rate) == self.sample_rate:
            return audio
        import torchaudio

        return torchaudio.functional.resample(audio, int(sample_rate), self.sample_rate)
        
    @staticmethod
    def _feature_tensor(output: Any) -> Tensor:
        """Extract projected features across Transformers CLAP API versions.

        Transformers 4.x returned a tensor from ``get_audio_features`` and
        ``get_text_features``.  Transformers 5.x returns a
        ``BaseModelOutputWithPooling`` whose normalized projected embedding is
        stored in ``pooler_output``.  Supporting both keeps evaluation
        notebooks independent of the locally installed Transformers version.
        """

        if isinstance(output, Tensor):
            return output
        for name in ("pooler_output", "audio_embeds", "text_embeds"):
            value = getattr(output, name, None)
            if isinstance(value, Tensor):
                return value
            if isinstance(output, Mapping):
                value = output.get(name)
                if isinstance(value, Tensor):
                    return value
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            return output[0]
        raise TypeError(
            "CLAP feature method returned no tensor or recognized embedding field"
        )
        
    @torch.inference_mode()
    def encode_audio(self, audio: Tensor, *, sample_rate: int, batch_size: int = 8) -> Tensor:
        waveforms = self._resample(audio, sample_rate)
        outputs = []
        for start in range(0, waveforms.shape[0], batch_size):
            values = [row.numpy() for row in waveforms[start : start + batch_size]]
            inputs = self.processor(
                audio=values,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self._feature_tensor(self.model.get_audio_features(**inputs))
            outputs.append(features.float().cpu())
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str], *, batch_size: int = 16) -> Tensor:
        outputs = []
        text_list = list(texts)
        for start in range(0, len(text_list), batch_size):
            inputs = self.processor(
                text=text_list[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self._feature_tensor(self.model.get_text_features(**inputs))
            outputs.append(features.float().cpu())
        return torch.cat(outputs, dim=0)

    def score(self, audio: Tensor, texts: Sequence[str], *, sample_rate: int) -> Tensor:
        audio_features = F.normalize(
            self.encode_audio(audio, sample_rate=sample_rate), dim=-1
        )
        text_features = F.normalize(self.encode_text(texts), dim=-1)
        return (audio_features * text_features).sum(dim=-1)


class MERTEncoder:
    """Mean-pooled MERT music embeddings for audio-audio comparisons."""

    def __init__(
        self,
        model_name_or_path: str = "m-a-p/MERT-v1-95M",
        *,
        device: torch.device | str = "cpu",
        local_files_only: bool = False,
        layer: int = -1,
    ) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.device = torch.device(device)
        self.layer = int(layer)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).eval().to(self.device)
        self.sample_rate = int(getattr(self.processor, "sampling_rate", 24_000))

    @staticmethod
    def _mono(audio: Tensor) -> Tensor:
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if audio.ndim != 2:
            raise ValueError("audio must have shape [B,T] or [B,C,T]")
        return audio.detach().float().cpu()

    def _resample(self, audio: Tensor, sample_rate: int) -> Tensor:
        audio = self._mono(audio)
        if int(sample_rate) == self.sample_rate:
            return audio
        import torchaudio

        return torchaudio.functional.resample(audio, int(sample_rate), self.sample_rate)

    @torch.inference_mode()
    def encode_audio(self, audio: Tensor, *, sample_rate: int, batch_size: int = 8) -> Tensor:
        waveforms = self._resample(audio, sample_rate)
        embeddings = []
        for start in range(0, waveforms.shape[0], batch_size):
            values = [row.numpy() for row in waveforms[start : start + batch_size]]
            inputs = self.processor(
                values,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = getattr(output, "hidden_states", None)
            hidden = hidden_states[self.layer] if hidden_states else output.last_hidden_state
            attention_mask = inputs.get("attention_mask")
            feature_mask = None
            mask_builder = getattr(self.model, "_get_feature_vector_attention_mask", None)
            if attention_mask is not None and callable(mask_builder):
                feature_mask = mask_builder(hidden.shape[1], attention_mask)
            if feature_mask is None:
                pooled = hidden.float().mean(dim=1)
            else:
                feature_mask = feature_mask.to(hidden).unsqueeze(-1)
                pooled = (hidden.float() * feature_mask).sum(dim=1) / feature_mask.sum(
                    dim=1
                ).clamp_min(1)
            embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0)

