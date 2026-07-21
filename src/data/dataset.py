"""PyTorch Lightning data module for tokenized JamendoMaxCaps subsets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.jmd_max_caps import (
    JamendoMaxCapsCollator,
    JamendoMaxCapsDataset,
    LabelVocabulary,
    build_artist_genre_matrix,
    build_label_vocabulary,
    load_token_manifest,
    select_records,
    split_records_by_track,
)


class JamendoMaxCapsDataModule(pl.LightningDataModule):
    """Build deterministic train/val/test loaders from an EnCodec token cache.

    Splits are made at track level and stratified per artist, so adjacent clips
    from the same song cannot leak across stages while each sufficiently large
    artist catalogue remains represented.  Build-time subsetting is applied
    before the split and does not modify the external subset or its token cache.
    """

    def __init__(
        self,
        subset_dir: str | Path,
        *,
        token_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        processor_name_or_path: str = "checkpoints/musicgen-small",
        tokenizer: Any | None = None,
        local_files_only: bool = True,
        concept_granularity: str = "artist",
        genre_field: str = "coarse_genres",
        artist_keys: Sequence[str] | None = None,
        genres: Sequence[str] | None = None,
        max_artists: int | None = None,
        max_songs_per_artist: int | None = None,
        max_clips_per_song: int | None = None,
        max_samples: int | None = None,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        seed: int = 42,
        train_batch_size: int = 4,
        eval_batch_size: int | None = None,
        num_workers: int = 4,
        persistent_workers: bool = True,
        eval_num_workers: int | None = None,
        eval_persistent_workers: bool | None = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        drop_last_train: bool = True,
        max_text_length: int = 128,
        max_audio_token_length: int | None = None,
        audio_pad_token_id: int = 2048,
        max_cached_shards: int = 2,
        validate_tokens: bool = True,
        balance_artists: bool = True,
    ) -> None:
        super().__init__()
        if train_batch_size <= 0 or (eval_batch_size is not None and eval_batch_size <= 0):
            raise ValueError("batch sizes must be positive")
        if num_workers < 0 or (eval_num_workers is not None and eval_num_workers < 0):
            raise ValueError("worker counts must be non-negative")
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        self.subset_dir = Path(subset_dir).expanduser().resolve()
        self.token_dir = (
            Path(token_dir).expanduser().resolve()
            if token_dir is not None
            else self.subset_dir / "encodec_musicgen_32khz_4cb"
        )
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else self.token_dir / "metadata" / "manifest.jsonl"
        )
        self.processor_name_or_path = processor_name_or_path
        self.tokenizer = tokenizer
        self.processor: Any | None = None
        self.local_files_only = local_files_only
        self.concept_granularity = concept_granularity
        self.genre_field = genre_field
        self.artist_keys = list(artist_keys) if artist_keys is not None else None
        self.genres = list(genres) if genres is not None else None
        self.max_artists = max_artists
        self.max_songs_per_artist = max_songs_per_artist
        self.max_clips_per_song = max_clips_per_song
        self.max_samples = max_samples
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = int(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size or train_batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.eval_num_workers = (
            num_workers if eval_num_workers is None else eval_num_workers
        )
        self.eval_persistent_workers = (
            persistent_workers
            if eval_persistent_workers is None
            else eval_persistent_workers
        )
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last_train = drop_last_train
        self.max_text_length = max_text_length
        self.max_audio_token_length = max_audio_token_length
        self.audio_pad_token_id = audio_pad_token_id
        self.max_cached_shards = max_cached_shards
        self.validate_tokens = validate_tokens
        self.balance_artists = bool(balance_artists)

        self.train_dataset: JamendoMaxCapsDataset | None = None
        self.val_dataset: JamendoMaxCapsDataset | None = None
        self.test_dataset: JamendoMaxCapsDataset | None = None
        self.vocabulary: LabelVocabulary | None = None
        self.artist_genre_matrix: torch.Tensor | None = None
        self.selection_stats: dict[str, int] = {}
        self._collator: JamendoMaxCapsCollator | None = None
        self.save_hyperparameters(ignore=["tokenizer"])

    def prepare_data(self) -> None:
        # Only download in Lightning's single prepare-data process. Local-only
        # runs defer loading to setup() and never contact the network.
        if self.tokenizer is None and not self.local_files_only:
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained(
                self.processor_name_or_path,
                local_files_only=False,
            )

    def _setup_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.processor_name_or_path,
            local_files_only=self.local_files_only,
        )
        self.tokenizer = self.processor.tokenizer

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_dataset is not None:
            return
        self._setup_tokenizer()
        records = load_token_manifest(self.manifest_path)
        selected = select_records(
            records,
            artist_keys=self.artist_keys,
            genres=self.genres,
            genre_field=self.genre_field,
            max_artists=self.max_artists,
            max_songs_per_artist=self.max_songs_per_artist,
            max_clips_per_song=self.max_clips_per_song,
            max_samples=self.max_samples,
            seed=self.seed,
        )
        splits = split_records_by_track(
            selected,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            seed=self.seed,
        )
        self.vocabulary = build_label_vocabulary(
            selected, genre_field=self.genre_field
        )
        self.artist_genre_matrix = build_artist_genre_matrix(
            selected,
            self.vocabulary,
            genre_field=self.genre_field,
        )
        dataset_kwargs = {
            "vocabulary": self.vocabulary,
            "concept_granularity": self.concept_granularity,
            "genre_field": self.genre_field,
            "max_cached_shards": self.max_cached_shards,
            "validate_tokens": self.validate_tokens,
        }
        self.train_dataset = JamendoMaxCapsDataset(
            self.token_dir, records=splits["train"], **dataset_kwargs
        )
        self.val_dataset = JamendoMaxCapsDataset(
            self.token_dir,
            records=splits["val"] or splits["train"][:1],
            **dataset_kwargs,
        )
        self.test_dataset = JamendoMaxCapsDataset(
            self.token_dir,
            records=splits["test"] or splits["val"] or splits["train"][:1],
            **dataset_kwargs,
        )
        self._collator = JamendoMaxCapsCollator(
            self.tokenizer,
            max_text_length=self.max_text_length,
            max_audio_token_length=self.max_audio_token_length,
            audio_pad_token_id=self.audio_pad_token_id,
        )
        self.selection_stats = {
            "all": len(records),
            "selected": len(selected),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "artists": len(self.vocabulary.artist_to_index),
            "genres": len(self.vocabulary.genre_to_index),
        }

    @property
    def num_artists(self) -> int:
        if self.vocabulary is None:
            raise RuntimeError("call setup() before reading num_artists")
        return len(self.vocabulary.artist_to_index)

    @property
    def num_genres(self) -> int:
        if self.vocabulary is None:
            raise RuntimeError("call setup() before reading num_genres")
        return len(self.vocabulary.genre_to_index)

    @property
    def num_concepts(self) -> int:
        return self.num_artists if self.concept_granularity == "artist" else self.num_genres

    def _loader(
        self,
        dataset: JamendoMaxCapsDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        persistent_workers: bool,
        balance_artists: bool = False,
    ) -> DataLoader:
        if self._collator is None:
            raise RuntimeError("call setup() before requesting a dataloader")
        generator = torch.Generator().manual_seed(self.seed)
        sampler = None
        if balance_artists:
            labels = torch.as_tensor(dataset.artist_labels, dtype=torch.long)
            counts = torch.bincount(labels, minlength=dataset.num_artists).clamp_min(1)
            sample_weights = counts[labels].reciprocal().double()
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle and sampler is None,
            "sampler": sampler,
            "drop_last": drop_last and len(dataset) >= batch_size,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers and num_workers > 0,
            "pin_memory": self.pin_memory,
            "collate_fn": self._collator,
            "generator": generator,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return self._loader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=self.drop_last_train,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            balance_artists=self.balance_artists,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return self._loader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return self._loader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


__all__ = ["JamendoMaxCapsDataModule"]
