"""Raw-audio datasets for transferred artist classification.

The training and ordinary validation splits are reconstructed from a
JamendoMaxCaps selection. An optional generated validation set measures
accuracy on ordinary (non-suppressed) model generations. Suppression outputs
remain available for separate post-training evaluation.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.jmd_max_caps import (
    LabelVocabulary,
    build_label_vocabulary,
    load_token_manifest,
    select_records,
    split_records_by_track,
)


GENERATED_VALIDATION_SAMPLE_TYPES = ("default",)
SOURCE_SELECTION_KEYS = (
    "subset_dir",
    "manifest_path",
    "artist_keys",
    "genres",
    "selected_coarse_genres",
    "genre_field",
    "max_artists",
    "max_songs_per_artist",
    "max_clips_per_song",
    "max_samples",
    "val_fraction",
    "test_fraction",
    "seed",
)


def _artist_key(record: Mapping[str, Any]) -> str:
    for key in ("artist_key", "artist_id", "artist_name"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("audio record has no artist identifier")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


class ArtistAudioDataset(Dataset[dict[str, Any]]):
    """Load mono waveforms and crop them to at most ``duration_seconds``."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        audio_root: str | Path,
        vocabulary: LabelVocabulary,
        sample_rate: int = 24_000,
        duration_seconds: float = 15.0,
        random_crop: bool = False,
    ) -> None:
        if not records:
            raise ValueError("ArtistAudioDataset requires at least one record")
        if sample_rate <= 0 or duration_seconds <= 0:
            raise ValueError("sample_rate and duration_seconds must be positive")
        self.records = [dict(record) for record in records]
        self.audio_root = Path(audio_root).expanduser().resolve()
        self.vocabulary = vocabulary
        self.sample_rate = int(sample_rate)
        self.max_samples = int(round(sample_rate * duration_seconds))
        self.random_crop = bool(random_crop)
        self.labels = [
            int(self.vocabulary.artist_to_index[_artist_key(record)])
            for record in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _audio_path(self, record: Mapping[str, Any]) -> Path:
        path = Path(str(record["audio"]))
        if path.is_absolute():
            resolved = path.expanduser().resolve()
        else:
            resolved = (self.audio_root / path).resolve()
            try:
                resolved.relative_to(self.audio_root)
            except ValueError as error:
                raise ValueError(f"audio path escapes its root: {path}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    def __getitem__(self, index: int) -> dict[str, Any]:
        import soundfile as sf
        import torchaudio

        record = self.records[index]
        samples, source_rate = sf.read(
            self._audio_path(record),
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(samples).transpose(0, 1).mean(dim=0)
        if int(source_rate) != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                int(source_rate),
                self.sample_rate,
            )
        if waveform.numel() > self.max_samples:
            available = waveform.numel() - self.max_samples
            if self.random_crop:
                start = int(torch.randint(available + 1, ()).item())
            else:
                start = available // 2
            waveform = waveform[start : start + self.max_samples]
        return {
            "waveform": waveform.contiguous(),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "metadata": {
                "audio": str(record["audio"]),
                "artist_key": _artist_key(record),
                "artist_name": str(record.get("artist_name", "")),
                "track_id": str(
                    record.get("track_id", record.get("source_track_id", ""))
                ),
                "sample_type": str(record.get("sample_type", "gt")),
            },
        }


class MERTAudioCollator:
    """Normalize and pad variable-length waveforms with MERT's processor."""

    def __init__(self, processor: Any, sample_rate: int) -> None:
        self.processor = processor
        self.sample_rate = int(sample_rate)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty audio batch")
        waveforms = [
            torch.as_tensor(sample["waveform"]).float().cpu().numpy()
            for sample in samples
        ]
        processed = self.processor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_values": torch.as_tensor(processed["input_values"]).float(),
            "attention_mask": torch.as_tensor(processed["attention_mask"]).long(),
            "labels": torch.stack(
                [torch.as_tensor(sample["label"]).long() for sample in samples]
            ),
            "metadata": [sample["metadata"] for sample in samples],
        }


class ArtistClassificationDataModule(pl.LightningDataModule):
    """Train on GT audio and validate on GT plus ordinary generations."""

    def __init__(
        self,
        subset_dir: str | Path | None = None,
        *,
        manifest_path: str | Path | None = None,
        generated_audio_dir: str | Path | None = None,
        require_generated_validation: bool = False,
        generated_validation_sample_types: Sequence[str] = ("default",),
        source_checkpoint_path: str | Path | None = None,
        source_config_path: str | Path | None = None,
        inherit_source_selection: bool = True,
        model_name_or_path: str = "m-a-p/MERT-v1-95M",
        processor: Any | None = None,
        local_files_only: bool = False,
        artist_keys: Sequence[str] | None = None,
        genres: Sequence[str] | None = None,
        selected_coarse_genres: Sequence[str] | None = None,
        genre_field: str = "coarse_genres",
        max_artists: int | None = None,
        max_songs_per_artist: int | None = None,
        max_clips_per_song: int | None = None,
        max_samples: int | None = None,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        seed: int = 0,
        sample_rate: int = 24_000,
        duration_seconds: float = 15.0,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 4,
        eval_num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        balance_artists: bool = True,
    ) -> None:
        super().__init__()
        if train_batch_size <= 0 or eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if num_workers < 0 or eval_num_workers < 0:
            raise ValueError("worker counts must be non-negative")
        self.subset_dir = (
            Path(subset_dir).expanduser().resolve() if subset_dir else None
        )
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve() if manifest_path else None
        )
        self.generated_audio_dir = (
            Path(generated_audio_dir).expanduser().resolve()
            if generated_audio_dir
            else None
        )
        self.require_generated_validation = bool(require_generated_validation)
        requested_sample_types = tuple(
            str(sample_type) for sample_type in generated_validation_sample_types
        )
        if not requested_sample_types:
            raise ValueError("generated_validation_sample_types cannot be empty")
        unknown_sample_types = sorted(
            set(requested_sample_types) - set(GENERATED_VALIDATION_SAMPLE_TYPES)
        )
        if unknown_sample_types:
            raise ValueError(
                "unsupported generated validation sample types: "
                f"{unknown_sample_types}; only ordinary default generation is "
                "validated during classifier training"
            )
        self.generated_validation_sample_types = requested_sample_types
        self.source_checkpoint_path = source_checkpoint_path
        self.source_config_path = source_config_path
        self.inherit_source_selection = bool(inherit_source_selection)
        self.model_name_or_path = str(model_name_or_path)
        self.local_files_only = bool(local_files_only)
        self.selection: dict[str, Any] = {
            "artist_keys": list(artist_keys) if artist_keys is not None else None,
            "genres": (
                [genres] if isinstance(genres, str) else list(genres)
                if genres is not None
                else None
            ),
            "selected_coarse_genres": (
                [selected_coarse_genres]
                if isinstance(selected_coarse_genres, str)
                else list(selected_coarse_genres)
                if selected_coarse_genres is not None
                else None
            ),
            "genre_field": genre_field,
            "max_artists": max_artists,
            "max_songs_per_artist": max_songs_per_artist,
            "max_clips_per_song": max_clips_per_song,
            "max_samples": max_samples,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "seed": int(seed),
        }
        self.sample_rate = int(sample_rate)
        self.duration_seconds = float(duration_seconds)
        self.train_batch_size = int(train_batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.num_workers = int(num_workers)
        self.eval_num_workers = int(eval_num_workers)
        self.persistent_workers = bool(persistent_workers)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.balance_artists = bool(balance_artists)

        self.processor: Any | None = processor
        self.collator: MERTAudioCollator | None = None
        self.vocabulary: LabelVocabulary | None = None
        self.train_dataset: ArtistAudioDataset | None = None
        self.val_dataset: ArtistAudioDataset | None = None
        self.test_dataset: ArtistAudioDataset | None = None
        self.generated_val_datasets: dict[str, ArtistAudioDataset] = {}
        self.validation_names: list[str] = ["gt"]
        self.selection_stats: dict[str, int] = {}
        self.save_hyperparameters(ignore=["processor"])

    def prepare_data(self) -> None:
        if self.processor is None and not self.local_files_only:
            from transformers import Wav2Vec2FeatureExtractor

            Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                local_files_only=False,
            )

    def _source_selection(self) -> dict[str, Any] | None:
        if not self.inherit_source_selection:
            return None
        if self.source_checkpoint_path or self.source_config_path:
            from omegaconf import OmegaConf
            from src.evaluation.checkpoint_runtime import locate_checkpoint_config

            if self.source_config_path:
                config_path = Path(self.source_config_path).expanduser().resolve()
            else:
                config_path = locate_checkpoint_config(self.source_checkpoint_path)
            config = OmegaConf.load(config_path)
            data = OmegaConf.to_container(config.runner.dataset, resolve=True)
            if not isinstance(data, dict):
                raise TypeError("source checkpoint dataset config must be a mapping")
            return data
        if self.generated_audio_dir is not None:
            metadata_path = self.generated_audio_dir / "run_metadata.json"
            if metadata_path.is_file():
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                data = payload.get("dataset")
                if isinstance(data, dict):
                    return data
        return None

    def _resolve_source(self) -> None:
        source = self._source_selection()
        if source:
            if source.get("subset_dir") is not None:
                self.subset_dir = Path(str(source["subset_dir"])).expanduser().resolve()
            if source.get("manifest_path") is not None:
                self.manifest_path = Path(
                    str(source["manifest_path"])
                ).expanduser().resolve()
            for key in SOURCE_SELECTION_KEYS:
                if key in {"subset_dir", "manifest_path"}:
                    continue
                if key in source:
                    self.selection[key] = source[key]
        if self.subset_dir is None:
            raise ValueError(
                "subset_dir is required unless it can be inherited from a "
                "source checkpoint/generated run"
            )
        if self.manifest_path is None:
            self.manifest_path = (
                self.subset_dir
                / "encodec_musicgen_32khz_4cb"
                / "metadata"
                / "manifest.jsonl"
            )

    def _setup_processor(self) -> None:
        if self.processor is None:
            from transformers import Wav2Vec2FeatureExtractor

            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        processor_rate = int(getattr(self.processor, "sampling_rate", self.sample_rate))
        if processor_rate != self.sample_rate:
            raise ValueError(
                f"classifier sample_rate={self.sample_rate} but processor expects "
                f"{processor_rate}"
            )
        self.collator = MERTAudioCollator(self.processor, self.sample_rate)

    def _generated_records(self) -> dict[str, list[dict[str, Any]]]:
        if self.generated_audio_dir is None:
            if self.require_generated_validation:
                raise ValueError(
                    "generated validation is required but generated_audio_dir "
                    "is unset; export CLASSIFIER_GENERATED_DIR or override "
                    "runner.dataset.require_generated_validation=false"
                )
            return {}
        manifest = self.generated_audio_dir / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        grouped = {
            sample_type: [] for sample_type in self.generated_validation_sample_types
        }
        for row in _load_jsonl(manifest):
            sample_type = str(row.get("sample_type", ""))
            if sample_type not in grouped:
                continue
            key = _artist_key(row)
            if key not in self.vocabulary.artist_to_index:
                raise ValueError(f"generated manifest contains unknown artist {key!r}")
            # ``artist_index`` in the generated manifest belongs to the
            # control checkpoint's concept vocabulary. A classifier may use a
            # larger class universe, so its label must be remapped by stable
            # artist_key rather than comparing unrelated integer namespaces.
            mapped = dict(row)
            mapped["control_artist_index"] = row.get("artist_index")
            mapped["classifier_artist_index"] = int(
                self.vocabulary.artist_to_index[key]
            )
            grouped[sample_type].append(mapped)
        available = {key: rows for key, rows in grouped.items() if rows}
        if self.require_generated_validation:
            missing_types = [
                sample_type
                for sample_type in self.generated_validation_sample_types
                if sample_type not in available
            ]
            if missing_types:
                raise ValueError(
                    "generated manifest is missing required sample types: "
                    f"{missing_types}"
                )
        return available

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_dataset is not None:
            return
        self._resolve_source()
        self._setup_processor()
        records = load_token_manifest(self.manifest_path)
        selected = select_records(
            records,
            artist_keys=self.selection["artist_keys"],
            genres=self.selection["genres"],
            selected_coarse_genres=self.selection["selected_coarse_genres"],
            genre_field=str(self.selection["genre_field"]),
            max_artists=self.selection["max_artists"],
            max_songs_per_artist=self.selection["max_songs_per_artist"],
            max_clips_per_song=self.selection["max_clips_per_song"],
            max_samples=self.selection["max_samples"],
            seed=int(self.selection["seed"]),
        )
        splits = split_records_by_track(
            selected,
            val_fraction=float(self.selection["val_fraction"]),
            test_fraction=float(self.selection["test_fraction"]),
            seed=int(self.selection["seed"]),
        )
        self.vocabulary = build_label_vocabulary(
            selected,
            genre_field=str(self.selection["genre_field"]),
        )
        common = {
            "audio_root": self.subset_dir,
            "vocabulary": self.vocabulary,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
        }
        self.train_dataset = ArtistAudioDataset(
            splits["train"], random_crop=True, **common
        )
        self.val_dataset = ArtistAudioDataset(
            splits["val"] or splits["train"][:1], random_crop=False, **common
        )
        self.test_dataset = ArtistAudioDataset(
            splits["test"] or splits["val"] or splits["train"][:1],
            random_crop=False,
            **common,
        )
        for sample_type, generated_records in self._generated_records().items():
            self.generated_val_datasets[sample_type] = ArtistAudioDataset(
                generated_records,
                audio_root=self.generated_audio_dir,
                vocabulary=self.vocabulary,
                sample_rate=self.sample_rate,
                duration_seconds=self.duration_seconds,
                random_crop=False,
            )
        self.validation_names = ["gt", *self.generated_val_datasets]
        self.selection_stats = {
            "selected": len(selected),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "artists": len(self.vocabulary.artist_to_index),
            **{
                f"generated_{key}": len(dataset)
                for key, dataset in self.generated_val_datasets.items()
            },
        }

    @property
    def num_classes(self) -> int:
        if self.vocabulary is None:
            raise RuntimeError("call setup() before reading num_classes")
        return len(self.vocabulary.artist_to_index)

    def _loader_kwargs(self, workers: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": self.pin_memory,
            "collate_fn": self.collator,
            "persistent_workers": self.persistent_workers and workers > 0,
        }
        if workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("call setup() before requesting a dataloader")
        sampler = None
        shuffle = True
        if self.balance_artists:
            counts = Counter(self.train_dataset.labels)
            weights = torch.tensor(
                [1.0 / counts[label] for label in self.train_dataset.labels],
                dtype=torch.double,
            )
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            **self._loader_kwargs(self.num_workers),
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("call setup() before requesting a dataloader")
        datasets = [self.val_dataset, *self.generated_val_datasets.values()]
        return [
            DataLoader(
                dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
                **self._loader_kwargs(self.eval_num_workers),
            )
            for dataset in datasets
        ]

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("call setup() before requesting a dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(self.eval_num_workers),
        )


__all__ = [
    "ArtistAudioDataset",
    "ArtistClassificationDataModule",
    "GENERATED_VALIDATION_SAMPLE_TYPES",
    "MERTAudioCollator",
]
