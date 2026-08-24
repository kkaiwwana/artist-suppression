"""Tracked audio comparisons for explicit MusicGen control probes."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from torch import Tensor

from src.experiment_logging import ExperimentTracker, experiment_tracker


log = logging.getLogger(__name__)


def _select_rows(value: Any, indices: list[int], batch_size: int) -> Any:
    if (
        isinstance(value, Tensor)
        and value.ndim > 0
        and value.shape[0] == batch_size
    ):
        row_indices = torch.as_tensor(indices, device=value.device)
        return value.index_select(0, row_indices)
    if isinstance(value, Mapping):
        return {
            key: _select_rows(item, indices, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, list) and len(value) == batch_size:
        return [value[index] for index in indices]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[index] for index in indices)
    return value


def _index_batch(
    batch: Mapping[str, Any],
    indices: list[int],
) -> dict[str, Any]:
    batch_size = len(batch["text"])
    return {
        key: _select_rows(value, indices, batch_size)
        for key, value in batch.items()
    }


def _slice_batch(batch: Mapping[str, Any], count: int) -> dict[str, Any]:
    return _index_batch(batch, list(range(min(count, len(batch["text"])))))


def _artist_identity(batch: Mapping[str, Any], index: int) -> str:
    labels = batch.get("artist_label")
    if isinstance(labels, Tensor) and labels.ndim > 0:
        return f"label:{int(labels[index])}"
    metadata = batch.get("metadata")
    if isinstance(metadata, (list, tuple)) and index < len(metadata):
        item = metadata[index]
        if isinstance(item, Mapping):
            value = item.get("artist_key") or item.get("artist_name")
            if value:
                return f"metadata:{value}"
    return ""


def _different_artist_batch(
    source: Mapping[str, Any],
    candidates: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Select one caption-B row with a different artist for every A row."""

    candidate_count = len(candidates["text"])
    if candidate_count == 0:
        return None
    candidate_identities = [
        _artist_identity(candidates, index) for index in range(candidate_count)
    ]
    selected: list[int] = []
    for source_index in range(len(source["text"])):
        source_identity = _artist_identity(source, source_index)
        matches = [
            index
            for index, candidate_identity in enumerate(candidate_identities)
            if source_identity
            and candidate_identity
            and candidate_identity != source_identity
        ]
        if not matches:
            return None
        selected.append(matches[source_index % len(matches)])
    return _index_batch(candidates, selected)


def _map_tensors(value: Any, function) -> Any:
    if isinstance(value, Tensor):
        return function(value)
    if isinstance(value, Mapping):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    return value


def _detach_to_cpu(batch: Mapping[str, Any]) -> dict[str, Any]:
    return _map_tensors(batch, lambda value: value.detach().cpu().clone())


def _move_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return _map_tensors(batch, lambda value: value.to(device))


class ValidationAudioComparisonCallback(pl.Callback):
    """Log fixed training-set probes after validation metrics are finalized.

    The historical class name is retained for configuration compatibility. The
    probe examples are captured once from training batches, kept on CPU, and
    queued at ``on_validation_end`` and generated at the next train-epoch
    boundary (or fit end). Long MusicGen generation therefore cannot interfere
    with Lightning's live validation metric accumulators or checkpoint scoring.
    """

    def __init__(
        self,
        *,
        every_n_epochs: int = 1,
        num_samples: int = 2,
        generation_seed: int = 2026,
        max_audio_seconds: float = 10.0,
        log_key: str = "training/audio_comparisons",
        fail_on_error: bool = False,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if every_n_epochs <= 0 or num_samples <= 0 or max_audio_seconds <= 0:
            raise ValueError("epoch/sample/audio limits must be positive")
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.generation_seed = generation_seed
        self.max_audio_seconds = max_audio_seconds
        self.log_key = log_key
        self.fail_on_error = fail_on_error
        self.generation_kwargs = dict(generation_kwargs or {})
        self._logged_epoch: int | None = None
        self._pending_epoch: int | None = None
        self._source_batch: dict[str, Any] | None = None
        self._counterfactual_batch: dict[str, Any] | None = None

    @staticmethod
    def _experiment_tracker(trainer: pl.Trainer) -> ExperimentTracker | None:
        return experiment_tracker(trainer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "source_batch": self._source_batch,
            "counterfactual_batch": self._counterfactual_batch,
            "pending_epoch": self._pending_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._source_batch = state_dict.get("source_batch")
        self._counterfactual_batch = state_dict.get("counterfactual_batch")
        self._pending_epoch = state_dict.get("pending_epoch")

    def _should_log(self, trainer: pl.Trainer) -> bool:
        return (
            trainer.is_global_zero
            and not trainer.sanity_checking
            and trainer.current_epoch % self.every_n_epochs == 0
            and self._logged_epoch != trainer.current_epoch
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> None:
        del pl_module
        del outputs
        del batch_idx
        if (
            not trainer.is_global_zero
            or trainer.sanity_checking
            or self._counterfactual_batch is not None
        ):
            return
        if "text" not in batch or not batch["text"]:
            return
        candidate_batch = _detach_to_cpu(batch)
        if self._source_batch is None:
            count = min(self.num_samples, len(candidate_batch["text"]))
            self._source_batch = _slice_batch(candidate_batch, count)
        counterfactual = _different_artist_batch(
            self._source_batch,
            candidate_batch,
        )
        if counterfactual is not None:
            self._counterfactual_batch = _detach_to_cpu(counterfactual)

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self._should_log(trainer):
            return
        # ModelCheckpoint callbacks are deliberately ordered last by Lightning.
        # Queue generation until the next train epoch so checkpoints and metric
        # buffers have already completed. The final epoch is flushed at fit end.
        self._pending_epoch = trainer.current_epoch

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._flush_pending(trainer, pl_module)

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._flush_pending(trainer, pl_module)

    def _flush_pending(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        epoch = self._pending_epoch
        if epoch is None or self._logged_epoch == epoch:
            return
        modules = getattr(pl_module, "modules", None)
        module_modes = (
            tuple((module, bool(module.training)) for module in modules())
            if callable(modules)
            else ()
        )
        root_was_training = bool(getattr(pl_module, "training", False))
        eval_model = getattr(pl_module, "eval", None)
        if callable(eval_model):
            eval_model()
        try:
            self._log_comparison(trainer, pl_module, epoch)
        finally:
            self._pending_epoch = None
            if module_modes:
                # Restore every submodule exactly. Calling pl_module.train()
                # here would recursively switch the frozen pretrained
                # MusicGen backbone from eval to train after epoch 0.
                for module, was_training in module_modes:
                    module.training = was_training
            else:
                train_model = getattr(pl_module, "train", None)
                if root_was_training and callable(train_model):
                    train_model()

    def _log_comparison(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> None:
        tracker = self._experiment_tracker(trainer)
        if tracker is None:
            log.warning(
                "Training audio comparison skipped: supported experiment "
                "logger not found"
            )
            self._logged_epoch = epoch
            return
        generate = getattr(pl_module, "generate_validation_comparisons", None)
        generate_default = getattr(pl_module, "generate_validation_samples", None)
        if not callable(generate) and not callable(generate_default):
            log.warning(
                "Training audio comparison skipped: model has neither "
                "generate_validation_comparisons() nor "
                "generate_validation_samples()"
            )
            self._logged_epoch = epoch
            return

        try:
            if self._source_batch is None:
                log.warning("Training audio comparison skipped: no fixed probe batch")
                self._logged_epoch = epoch
                return
            if not callable(generate):
                self._log_default_samples(
                    tracker,
                    trainer,
                    pl_module,
                    epoch,
                    generate_default,
                    tracker.media,
                )
                return
            if self._counterfactual_batch is None:
                log.warning(
                    "Training audio comparison skipped: no fixed different-artist "
                    "probe pair was captured"
                )
                self._logged_epoch = epoch
                return
            device = pl_module.device
            source_batch = _move_to_device(self._source_batch, device)
            counterfactual_batch = _move_to_device(
                self._counterfactual_batch,
                device,
            )

            count = len(source_batch["text"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            comparison = generate(
                source_batch,
                counterfactual_batch=counterfactual_batch,
                generation_kwargs=self.generation_kwargs,
                generation_seed=self.generation_seed,
            )
            sample_rate = int(comparison["sample_rate"])
            max_samples = int(round(self.max_audio_seconds * sample_rate))
            prompts = list(comparison.get("prompts") or source_batch["text"])
            other_prompts = list(
                comparison.get("other_prompts")
                or counterfactual_batch["text"]
            )
            metadata = comparison.get("metadata") or [{} for _ in range(count)]
            other_metadata = comparison.get("other_metadata") or [
                {} for _ in range(count)
            ]
            ground_truth = comparison["ground_truth"].detach().float().cpu()
            ground_lengths = comparison["ground_truth_lengths"].detach().cpu()
            generated = comparison["generated"].detach().float().cpu()
            positive = comparison["positive"].detach().float().cpu()
            suppressed = comparison["suppressed"].detach().float().cpu()
            other_ground_truth = (
                comparison["other_ground_truth"].detach().float().cpu()
            )
            other_ground_lengths = comparison[
                "other_ground_truth_lengths"
            ].detach().cpu()
            other_generated = (
                comparison["other_generated"].detach().float().cpu()
            )
            suppressed_with_other_caption = comparison[
                "suppressed_with_other_caption"
            ].detach().float().cpu()
            media = tracker.media
            table = media.Table(
                columns=[
                    "epoch",
                    "sample",
                    "artist_A",
                    "genres_A",
                    "caption_A",
                    "ground_truth",
                    "default_generated",
                    "positive_artist_A",
                    "suppressed_A_caption_A",
                    "artist_B",
                    "caption_B",
                    "ground_truth_B",
                    "generated_B",
                    "suppressed_A_caption_B",
                ]
            )
            for index in range(count):
                item_metadata = metadata[index] if index < len(metadata) else {}
                other_item_metadata = (
                    other_metadata[index] if index < len(other_metadata) else {}
                )
                gt_length = min(int(ground_lengths[index]), max_samples)
                gen_length = min(generated.shape[-1], max_samples)
                positive_length = min(positive.shape[-1], max_samples)
                suppressed_length = min(suppressed.shape[-1], max_samples)
                counterfactual_length = min(
                    suppressed_with_other_caption.shape[-1], max_samples
                )
                other_gt_length = min(
                    int(other_ground_lengths[index]), max_samples
                )
                other_gen_length = min(other_generated.shape[-1], max_samples)
                table.add_data(
                    epoch,
                    index,
                    item_metadata.get(
                        "artist_name", item_metadata.get("artist_key", "")
                    ),
                    ", ".join(item_metadata.get("genres", [])),
                    prompts[index],
                    media.Audio(
                        ground_truth[index, 0, :gt_length].numpy(),
                        sample_rate=sample_rate,
                        caption="ground truth",
                    ),
                    media.Audio(
                        generated[index, 0, :gen_length].numpy(),
                        sample_rate=sample_rate,
                        caption="default generation (no artist control)",
                    ),
                    media.Audio(
                        positive[index, 0, :positive_length].numpy(),
                        sample_rate=sample_rate,
                        caption=f"positive artist A: {prompts[index]}",
                    ),
                    media.Audio(
                        suppressed[index, 0, :suppressed_length].numpy(),
                        sample_rate=sample_rate,
                        caption=f"suppress A with caption A: {prompts[index]}",
                    ),
                    other_item_metadata.get(
                        "artist_name", other_item_metadata.get("artist_key", "")
                    ),
                    other_prompts[index],
                    media.Audio(
                        other_ground_truth[index, 0, :other_gt_length].numpy(),
                        sample_rate=sample_rate,
                        caption="ground truth B",
                    ),
                    media.Audio(
                        other_generated[index, 0, :other_gen_length].numpy(),
                        sample_rate=sample_rate,
                        caption="normal generation B",
                    ),
                    media.Audio(
                        suppressed_with_other_caption[
                            index, 0, :counterfactual_length
                        ].numpy(),
                        sample_rate=sample_rate,
                        caption=(
                            "suppress artist A; generate from unrelated caption B: "
                            f"{other_prompts[index]}"
                        ),
                    ),
                )
            # Keep the Lightning step as data instead of forcing the tracker's
            # internal history step. Several logger payloads may be emitted per
            # optimization step, and backends require monotonic history steps.
            tracker.log(
                {
                    self.log_key: table,
                    "trainer/global_step": trainer.global_step,
                }
            )
            self._logged_epoch = epoch
        except Exception:  # noqa: BLE001 - optional monitor should be configurable
            if self.fail_on_error:
                raise
            log.exception("Training audio comparison failed; training continues")
            self._logged_epoch = epoch
        finally:
            device = getattr(pl_module, "device", torch.device("cpu"))
            if getattr(device, "type", None) == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

    def _log_default_samples(
        self,
        tracker: ExperimentTracker,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
        generate: Any,
        media_module: Any,
    ) -> None:
        """Log the compact GT/default table used by adapter-only training."""

        if self._source_batch is None:
            raise RuntimeError("a fixed source batch is required")
        device = pl_module.device
        source_batch = _move_to_device(self._source_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = generate(
            source_batch,
            generation_kwargs=self.generation_kwargs,
            generation_seed=self.generation_seed,
        )
        sample_rate = int(result["sample_rate"])
        max_samples = int(round(self.max_audio_seconds * sample_rate))
        prompts = list(result.get("prompts") or source_batch["text"])
        count = len(prompts)
        metadata = result.get("metadata") or [{} for _ in range(count)]
        ground_truth = result["ground_truth"].detach().float().cpu()
        ground_lengths = result["ground_truth_lengths"].detach().cpu()
        generated = result["generated"].detach().float().cpu()
        table = media_module.Table(
            columns=[
                "epoch",
                "sample",
                "artist",
                "genres",
                "caption",
                "ground_truth",
                "default_generated",
            ]
        )
        for index in range(count):
            item_metadata = metadata[index] if index < len(metadata) else {}
            gt_length = min(int(ground_lengths[index]), max_samples)
            generated_length = min(generated.shape[-1], max_samples)
            table.add_data(
                epoch,
                index,
                item_metadata.get(
                    "artist_name", item_metadata.get("artist_key", "")
                ),
                ", ".join(item_metadata.get("genres", [])),
                prompts[index],
                media_module.Audio(
                    ground_truth[index, 0, :gt_length].numpy(),
                    sample_rate=sample_rate,
                    caption="ground truth",
                ),
                media_module.Audio(
                    generated[index, 0, :generated_length].numpy(),
                    sample_rate=sample_rate,
                    caption="adapter-only default generation",
                ),
            )
        tracker.log(
            {
                self.log_key: table,
                "trainer/global_step": trainer.global_step,
            }
        )
        self._logged_epoch = epoch


__all__ = ["ValidationAudioComparisonCallback"]
