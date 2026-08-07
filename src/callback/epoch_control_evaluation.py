"""Fixed-cohort, six-scenario epoch-end control evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytorch_lightning as pl
import torch
from torch import Tensor

from src.evaluation.control_scenarios import (
    EVALUATION_METRICS,
    METRIC_LABELS,
    SCENARIO_NAMES,
    CohortItem,
    ScenarioCondition,
    attach_audio_prompt,
    audio_frame_rate,
    build_control_scenarios,
    fixed_audio_segment,
    formatted_table_rows,
    select_balanced_cohort,
    summarize_scenario_metrics,
)
from src.evaluation.metric_runtimes import (
    classifier_target_statistics,
    cosine_similarity_rows,
    frechet_distance_from_embeddings,
)
from src.metric.forgetting import (
    ground_truth_next_token_confidence_from_token_losses_per_sample,
    ground_truth_next_token_confidence_per_sample,
)


log = logging.getLogger(__name__)


def _map_tensors(value: Any, function: Callable[[Tensor], Tensor]) -> Any:
    if isinstance(value, Tensor):
        return function(value)
    if isinstance(value, Mapping):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    return value


def _move_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return _map_tensors(batch, lambda value: value.to(device))


def _runtime_tensor(value: Any, *, key: str = "kld") -> Tensor:
    if isinstance(value, Mapping):
        if key not in value:
            raise KeyError(f"metric result has no {key!r} value")
        value = value[key]
    return torch.as_tensor(value).detach().float().cpu().flatten()


def _passt_per_sample(
    runtime: Any,
    generated: Tensor,
    reference: Tensor,
    *,
    sample_rate: int,
    batch_size: int | None = None,
) -> Tensor:
    """Adapt PaSST runtimes while keeping interface drift in one place."""

    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("PaSST batch_size must be positive")
        if generated.shape[0] > batch_size:
            return torch.cat(
                [
                    _passt_per_sample(
                        runtime,
                        generated[start : start + batch_size],
                        reference[start : start + batch_size],
                        sample_rate=sample_rate,
                    )
                    for start in range(0, generated.shape[0], batch_size)
                ]
            )

    per_sample = getattr(runtime, "per_sample", None)
    if callable(per_sample):
        return _runtime_tensor(
            per_sample(generated, reference, sample_rate=sample_rate)
        )
    pairwise_scores = getattr(runtime, "pairwise_scores", None)
    if callable(pairwise_scores):
        result = pairwise_scores(generated, reference, sample_rate=sample_rate)
        values = _runtime_tensor(result, key="pq")
        if values.numel() != generated.shape[0]:
            raise ValueError(
                "PaSST pairwise_scores()['pq'] must contain one value per clip"
            )
        return values
    for method_name in ("pairwise", "pairwise_kl", "score"):
        method = getattr(runtime, method_name, None)
        if callable(method):
            result = method(generated, reference, sample_rate=sample_rate)
            values = _runtime_tensor(result)
            if values.numel() == generated.shape[0]:
                return values

    # TorchMetric's public aggregate API can still produce exact per-clip
    # values by evaluating one pair at a time. This is slower, but preserves
    # the requested population standard deviation without private attributes.
    update = getattr(runtime, "update", None)
    compute = getattr(runtime, "compute", None)
    reset = getattr(runtime, "reset", None)
    if not all(callable(method) for method in (update, compute, reset)):
        raise TypeError(
            "PaSSTKLDivergence must provide per_sample(), pairwise(), score(), "
            "or the TorchMetric update/compute/reset API"
        )
    values: list[Tensor] = []
    for index in range(generated.shape[0]):
        reset()
        update(
            generated[index : index + 1],
            reference[index : index + 1],
            sample_rate=sample_rate,
        )
        item = _runtime_tensor(compute())
        if item.numel() != 1:
            raise ValueError("single-pair PaSST compute() must return one kld value")
        values.append(item)
    reset()
    return torch.cat(values)


def _vggish_embeddings(runtime: Any, audio: Tensor, *, sample_rate: int) -> Tensor:
    """Return all VGGish window embeddings for an audio group."""

    for method_name in ("encode_audio", "forward"):
        method = getattr(runtime, method_name, None)
        if callable(method):
            try:
                output = method(audio, sample_rate=sample_rate)
            except TypeError:
                output = method(audio, sample_rate)
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, list):
                if not output:
                    raise ValueError("VGGish returned an empty embedding list")
                windows = [
                    torch.as_tensor(item).detach().float().cpu() for item in output
                ]
                if any(item.ndim != 2 for item in windows):
                    raise ValueError(
                        "every VGGish clip embedding must have shape [windows, D]"
                    )
                output = torch.cat(windows, dim=0)
            embeddings = torch.as_tensor(output).detach().float().cpu()
            if embeddings.ndim != 2:
                raise ValueError("VGGish embeddings must have shape [windows, D]")
            return embeddings
    if callable(runtime):
        try:
            output = runtime(audio, sample_rate=sample_rate)
        except TypeError:
            output = runtime(audio, sample_rate)
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, list):
            if not output:
                raise ValueError("VGGish returned an empty embedding list")
            windows = [torch.as_tensor(item).detach().float().cpu() for item in output]
            if any(item.ndim != 2 for item in windows):
                raise ValueError(
                    "every VGGish clip embedding must have shape [windows, D]"
                )
            output = torch.cat(windows, dim=0)
        embeddings = torch.as_tensor(output).detach().float().cpu()
        if embeddings.ndim != 2:
            raise ValueError("VGGish embeddings must have shape [windows, D]")
        return embeddings
    raise TypeError("VGGishAudioEmbedding must be callable or provide encode_audio()")


def _release_runtime(runtime: Any) -> None:
    model = getattr(runtime, "model", None)
    if model is not None and callable(getattr(model, "to", None)):
        model.to("cpu")
    elif callable(getattr(runtime, "to", None)):
        runtime.to("cpu")
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_runtime(runtime: Any, device: torch.device) -> Any:
    """Move a metric runtime without requiring every wrapper to define to()."""

    move = getattr(runtime, "to", None)
    if callable(move):
        moved = move(device)
        return runtime if moved is None else moved
    model = getattr(runtime, "model", None)
    if model is not None and callable(getattr(model, "to", None)):
        model.to(device)
    if hasattr(runtime, "device"):
        runtime.device = device
    return runtime


@dataclass(frozen=True)
class EpochControlArtifacts:
    """CPU results produced before loading any external evaluator."""

    epoch: int
    sample_rate: int
    texts: tuple[str, ...]
    artist_keys: tuple[str, ...]
    artist_ids: Tensor
    reference_audio: Tensor
    generated_audio: Mapping[str, Tensor]
    gt_token_confidence: Mapping[str, Tensor]


class EpochControlEvaluationCallback(pl.Callback):
    """Evaluate six matched controls on a fixed balanced training cohort.

    Evaluation is queued after validation and flushed at the next train epoch
    boundary (or fit end), after checkpoint callbacks have consumed validation
    metrics. The cohort and multi-artist sets stay fixed for the full run.
    """

    def __init__(
        self,
        *,
        every_n_epochs: int = 1,
        num_artists: int = 4,
        clips_per_artist: int = 16,
        generation_batch_size: int = 16,
        metric_batch_size: int = 8,
        generation_seed: int = 2026,
        cohort_seed: int = 42,
        audio_prompt_seconds: float = 5.0,
        continuation_seconds: float = 10.0,
        multi_min_artists: int = 2,
        multi_max_artists: int = 5,
        classifier_checkpoint_path: str | Path | None = None,
        classifier_config_path: str | Path | None = None,
        classifier_model_name_or_path: str | Path | None = None,
        mert_model_name_or_path: str = "m-a-p/MERT-v1-95M",
        clap_model_name_or_path: str = "laion/clap-htsat-unfused",
        local_files_only: bool = False,
        metric_device: str | torch.device | None = None,
        log_key: str = "evaluation/control_metrics",
        numeric_namespace: str = "evaluation/control_metric_stats",
        fail_on_error: bool = False,
        generation_kwargs: Mapping[str, Any] | None = None,
        runtime_factories: Mapping[str, Callable[[], Any]] | None = None,
        evaluation_runner: (
            Callable[[EpochControlArtifacts], Mapping[str, Mapping[str, Any]]] | None
        ) = None,
    ) -> None:
        super().__init__()
        positive_ints = {
            "every_n_epochs": every_n_epochs,
            "num_artists": num_artists,
            "clips_per_artist": clips_per_artist,
            "generation_batch_size": generation_batch_size,
            "metric_batch_size": metric_batch_size,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"positive values required for: {', '.join(invalid)}")
        if audio_prompt_seconds <= 0 or continuation_seconds <= 0:
            raise ValueError("audio prompt and continuation durations must be positive")
        if multi_min_artists < 2 or multi_max_artists < multi_min_artists:
            raise ValueError("multi bounds must satisfy 2 <= min <= max")

        self.every_n_epochs = int(every_n_epochs)
        self.num_artists = int(num_artists)
        self.clips_per_artist = int(clips_per_artist)
        self.generation_batch_size = int(generation_batch_size)
        self.metric_batch_size = int(metric_batch_size)
        self.generation_seed = int(generation_seed)
        self.cohort_seed = int(cohort_seed)
        self.audio_prompt_seconds = float(audio_prompt_seconds)
        self.continuation_seconds = float(continuation_seconds)
        self.multi_min_artists = int(multi_min_artists)
        self.multi_max_artists = int(multi_max_artists)
        self.classifier_checkpoint_path = (
            None
            if classifier_checkpoint_path is None
            else str(classifier_checkpoint_path)
        )
        self.classifier_config_path = (
            None if classifier_config_path is None else str(classifier_config_path)
        )
        self.classifier_model_name_or_path = (
            None
            if classifier_model_name_or_path is None
            else str(classifier_model_name_or_path)
        )
        self.mert_model_name_or_path = str(mert_model_name_or_path)
        self.clap_model_name_or_path = str(clap_model_name_or_path)
        self.local_files_only = bool(local_files_only)
        self.metric_device = (
            None if metric_device is None else torch.device(metric_device)
        )
        self.log_key = str(log_key)
        self.numeric_namespace = str(numeric_namespace).rstrip("/")
        self.fail_on_error = bool(fail_on_error)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.runtime_factories = dict(runtime_factories or {})
        self.evaluation_runner = evaluation_runner

        self._cohort: tuple[CohortItem, ...] | None = None
        self._scenario_plan: tuple[ScenarioCondition, ...] | None = None
        self._pending_epoch: int | None = None
        self._logged_epoch: int | None = None

    @staticmethod
    def _wandb_experiment(trainer: pl.Trainer) -> Any | None:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:  # pragma: no cover
            return None
        loggers = getattr(trainer, "loggers", None) or [
            getattr(trainer, "logger", None)
        ]
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                return logger.experiment
        return None

    def state_dict(self) -> dict[str, Any]:
        cohort = None
        if self._cohort is not None:
            cohort = [
                {
                    "dataset_index": item.dataset_index,
                    "artist_key": item.artist_key,
                    "artist_index": item.artist_index,
                }
                for item in self._cohort
            ]
        return {
            "cohort": cohort,
            "pending_epoch": self._pending_epoch,
            "logged_epoch": self._logged_epoch,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        cohort = state_dict.get("cohort")
        self._cohort = (
            None
            if cohort is None
            else tuple(CohortItem(**dict(item)) for item in cohort)
        )
        self._scenario_plan = None
        self._pending_epoch = state_dict.get("pending_epoch")
        self._logged_epoch = state_dict.get("logged_epoch")

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del pl_module
        if (
            not trainer.sanity_checking
            and (int(trainer.current_epoch) + 1) % self.every_n_epochs == 0
            and self._logged_epoch != int(trainer.current_epoch)
        ):
            self._pending_epoch = int(trainer.current_epoch)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._flush_pending(trainer, pl_module)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._flush_pending(trainer, pl_module)

    @staticmethod
    def _barrier(trainer: pl.Trainer, name: str) -> None:
        """Keep nonzero ranks from entering the next DDP step during eval."""

        barrier = getattr(getattr(trainer, "strategy", None), "barrier", None)
        if not callable(barrier):
            return
        try:
            barrier(name)
        except TypeError:  # compatibility with small/custom strategies
            barrier()

    def _flush_pending(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = self._pending_epoch
        if epoch is None or self._logged_epoch == epoch:
            return
        self._barrier(trainer, "epoch-control-evaluation-start")
        if not trainer.is_global_zero:
            # Rank zero performs all generation and external-model inference;
            # the matching barrier makes every other rank wait before the next
            # training collective instead of timing out in DDP.
            self._pending_epoch = None
            self._logged_epoch = epoch
            self._barrier(trainer, "epoch-control-evaluation-end")
            return
        modules = getattr(pl_module, "modules", None)
        module_modes = (
            tuple((module, bool(module.training)) for module in modules())
            if callable(modules)
            else ()
        )
        root_was_training = bool(getattr(pl_module, "training", False))
        if callable(getattr(pl_module, "eval", None)):
            pl_module.eval()
        try:
            self._run_and_log(trainer, pl_module, epoch)
        except Exception:  # noqa: BLE001 - this is an optional expensive monitor
            self._logged_epoch = epoch
            if self.fail_on_error:
                raise
            log.exception("Epoch control evaluation failed; training continues")
        finally:
            self._pending_epoch = None
            self._logged_epoch = epoch
            if module_modes:
                for module, was_training in module_modes:
                    module.training = was_training
            elif root_was_training and callable(getattr(pl_module, "train", None)):
                pl_module.train()
            device = torch.device(getattr(pl_module, "device", "cpu"))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            self._barrier(trainer, "epoch-control-evaluation-end")

    @staticmethod
    def _training_data(
        trainer: pl.Trainer,
    ) -> tuple[Any, Sequence[Mapping[str, Any]], Mapping[str, int], Callable]:
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("epoch control evaluation requires trainer.datamodule")
        dataset = getattr(datamodule, "train_dataset", None)
        records = getattr(dataset, "records", None)
        if records is None:
            records = getattr(datamodule, "records", None)
        if dataset is None:
            dataset = records
        vocabulary = getattr(datamodule, "vocabulary", None)
        if vocabulary is None and dataset is not None:
            vocabulary = getattr(dataset, "vocabulary", None)
        artist_to_index = getattr(vocabulary, "artist_to_index", vocabulary)
        collator = getattr(datamodule, "collator", None) or getattr(
            datamodule, "_collator", None
        )
        if (
            dataset is None
            or records is None
            or not isinstance(artist_to_index, Mapping)
        ):
            raise RuntimeError(
                "datamodule must expose train records and an artist vocabulary"
            )
        if not callable(collator):
            raise RuntimeError("datamodule must expose a callable collator/_collator")
        return dataset, records, artist_to_index, collator

    def _ensure_cohort(
        self,
        records: Sequence[Mapping[str, Any]],
        artist_to_index: Mapping[str, int],
        *,
        minimum_token_frames: int,
        allowed_artist_keys: set[str] | None,
    ) -> tuple[CohortItem, ...]:
        if self._cohort is None:
            self._cohort = select_balanced_cohort(
                records,
                artist_to_index,
                num_artists=self.num_artists,
                clips_per_artist=self.clips_per_artist,
                seed=self.cohort_seed,
                allowed_artist_keys=allowed_artist_keys,
                minimum_token_frames=minimum_token_frames,
            )
        if self._scenario_plan is None:
            target_ids = torch.tensor(
                [item.artist_index for item in self._cohort], dtype=torch.long
            )
            self._scenario_plan = build_control_scenarios(
                target_ids,
                num_concepts=len(artist_to_index),
                seed=self.generation_seed,
                multi_min_artists=self.multi_min_artists,
                multi_max_artists=self.multi_max_artists,
            )
        return self._cohort

    def _classifier_vocabulary_filter(self) -> set[str] | None:
        """Read checkpoint metadata before sampling classifiable artists."""

        if self.evaluation_runner is not None or "classifier" in self.runtime_factories:
            return None
        if not self.classifier_checkpoint_path:
            return None
        from src.evaluation.checkpoint_runtime import trusted_torch_load

        payload = trusted_torch_load(self.classifier_checkpoint_path)
        if not isinstance(payload, Mapping):
            raise ValueError("artist classifier checkpoint must contain a mapping")
        vocabulary = payload.get("artist_vocabulary")
        if not isinstance(vocabulary, Sequence) or isinstance(vocabulary, (str, bytes)):
            raise ValueError(
                "artist classifier checkpoint has no artist_vocabulary metadata"
            )
        artist_keys = {str(value) for value in vocabulary}
        if not artist_keys:
            raise ValueError("artist classifier checkpoint has an empty vocabulary")
        return artist_keys

    @staticmethod
    def _slice_scenario(
        scenario: ScenarioCondition,
        start: int,
        stop: int,
        device: torch.device,
    ) -> ScenarioCondition:
        return ScenarioCondition(
            name=scenario.name,
            direction=scenario.direction,
            weights=(
                None
                if scenario.weights is None
                else scenario.weights[start:stop].to(device)
            ),
            artist_sets=scenario.artist_sets[start:stop],
        )

    @staticmethod
    def _condition(pl_module: Any, scenario: ScenarioCondition) -> Any | None:
        if scenario.weights is None:
            return None
        learner = getattr(pl_module, "concept_learner", None)
        prepare = getattr(learner, "prepare_condition", None)
        if not callable(prepare):
            raise RuntimeError("model must expose concept_learner.prepare_condition()")
        return prepare(concept_weights=scenario.weights, direction=scenario.direction)

    def _collect_artifacts(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> EpochControlArtifacts:
        dataset, records, artist_to_index, collator = self._training_data(trainer)
        device = torch.device(getattr(pl_module, "device", "cpu"))
        frame_rate = float(audio_frame_rate(pl_module))
        prompt_frames = max(1, int(round(self.audio_prompt_seconds * frame_rate)))
        continuation_frames = max(1, int(round(self.continuation_seconds * frame_rate)))
        encoded_prompt_seconds = prompt_frames / frame_rate
        cohort = self._ensure_cohort(
            records,
            artist_to_index,
            minimum_token_frames=prompt_frames + continuation_frames,
            allowed_artist_keys=self._classifier_vocabulary_filter(),
        )
        assert self._scenario_plan is not None
        sample_rate = int(
            getattr(getattr(pl_module, "model", None), "audio_sample_rate", 32_000)
        )
        num_codebooks = int(
            getattr(getattr(pl_module, "model", None), "num_codebooks", 1) or 1
        )
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.setdefault(
            "max_new_tokens",
            int(math.ceil(self.continuation_seconds * frame_rate)) + num_codebooks,
        )

        generated: dict[str, list[Tensor]] = {name: [] for name in SCENARIO_NAMES}
        confidences: dict[str, list[Tensor]] = {name: [] for name in SCENARIO_NAMES}
        references: list[Tensor] = []
        texts: list[str] = []
        artist_keys: list[str] = []
        artist_ids: list[Tensor] = []
        decode = getattr(getattr(pl_module, "model", None), "decode_audio_tokens", None)
        run_generator = getattr(pl_module, "_run_generator", None)
        generate = getattr(pl_module, "_generate_with_condition", None)
        if not all(callable(method) for method in (decode, run_generator, generate)):
            raise RuntimeError(
                "model must expose decode_audio_tokens(), _run_generator(), and "
                "_generate_with_condition()"
            )

        for start in range(0, len(cohort), self.generation_batch_size):
            stop = min(start + self.generation_batch_size, len(cohort))
            items = cohort[start:stop]
            cpu_batch = collator([dataset[item.dataset_index] for item in items])
            batch = _move_to_device(cpu_batch, device)
            tokens = batch.get("audio_tokens")
            decoder_mask = batch.get("decoder_attention_mask")
            if not isinstance(tokens, Tensor) or not isinstance(decoder_mask, Tensor):
                raise ValueError(
                    "collator must return audio_tokens and decoder_attention_mask"
                )
            reference_audio, _ = decode(tokens, decoder_mask)
            references.append(
                fixed_audio_segment(
                    reference_audio,
                    sample_rate=sample_rate,
                    start_seconds=encoded_prompt_seconds,
                    duration_seconds=self.continuation_seconds,
                )
                .detach()
                .float()
                .cpu()
            )
            prompted_batch = attach_audio_prompt(batch, prompt_frames=prompt_frames)
            texts.extend(str(value) for value in batch.get("text", ()))
            artist_keys.extend(item.artist_key for item in items)
            artist_ids.append(
                torch.tensor([item.artist_index for item in items], dtype=torch.long)
            )

            batch_seed = self.generation_seed + start
            cuda_devices: list[int] = []
            if device.type == "cuda":
                cuda_devices = [
                    (
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    )
                ]
            for full_scenario in self._scenario_plan:
                scenario = self._slice_scenario(full_scenario, start, stop, device)
                condition = self._condition(pl_module, scenario)
                teacher_forced = run_generator(
                    batch,
                    start // self.generation_batch_size,
                    stage="val",
                    condition=condition,
                )
                logits = teacher_forced["logits"]
                labels = teacher_forced["labels"]
                token_losses = teacher_forced.get("token_losses")
                time = torch.arange(labels.shape[-1], device=labels.device)
                time_mask = (time >= prompt_frames) & (
                    time < prompt_frames + continuation_frames
                )
                time_mask = time_mask.unsqueeze(0).expand(labels.shape[0], -1)
                pad_token_id = int(
                    getattr(
                        getattr(pl_module, "model", None), "audio_pad_token_id", 2048
                    )
                )
                if isinstance(token_losses, Tensor):
                    gt_confidence = (
                        ground_truth_next_token_confidence_from_token_losses_per_sample(
                            token_losses,
                            labels,
                            pad_token_id=pad_token_id,
                            time_mask=time_mask,
                        )
                    )
                else:
                    gt_confidence = ground_truth_next_token_confidence_per_sample(
                        logits,
                        labels,
                        pad_token_id=pad_token_id,
                        time_mask=time_mask,
                    )
                confidences[scenario.name].append(gt_confidence.cpu())
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(batch_seed)
                    generated_audio = generate(
                        prompted_batch, generation_kwargs, condition
                    )
                generated[scenario.name].append(
                    fixed_audio_segment(
                        generated_audio,
                        sample_rate=sample_rate,
                        start_seconds=encoded_prompt_seconds,
                        duration_seconds=self.continuation_seconds,
                    )
                    .detach()
                    .float()
                    .cpu()
                )

        return EpochControlArtifacts(
            epoch=epoch,
            sample_rate=sample_rate,
            texts=tuple(texts),
            artist_keys=tuple(artist_keys),
            artist_ids=torch.cat(artist_ids),
            reference_audio=torch.cat(references),
            generated_audio={
                name: torch.cat(generated[name]) for name in SCENARIO_NAMES
            },
            gt_token_confidence={
                name: torch.cat(confidences[name]) for name in SCENARIO_NAMES
            },
        )

    def _runtime(self, name: str, default_factory: Callable[[], Any]) -> Any:
        factory = self.runtime_factories.get(name, default_factory)
        return factory()

    def _metric_runtime_device(self, pl_module: pl.LightningModule) -> torch.device:
        return self.metric_device or torch.device(getattr(pl_module, "device", "cpu"))

    def _evaluate_default(
        self,
        artifacts: EpochControlArtifacts,
        *,
        device: torch.device,
    ) -> dict[str, dict[str, Tensor]]:
        values: dict[str, dict[str, Tensor]] = {
            name: {"gt_token_confidence": artifacts.gt_token_confidence[name]}
            for name in SCENARIO_NAMES
        }

        def classifier_factory() -> Any:
            if not self.classifier_checkpoint_path:
                raise RuntimeError(
                    "classifier_checkpoint_path is required for epoch control evaluation"
                )
            from src.evaluation.metric_runtimes import ArtistClassifierRuntime

            return ArtistClassifierRuntime.from_checkpoint(
                self.classifier_checkpoint_path,
                config_path=self.classifier_config_path,
                model_name_or_path=self.classifier_model_name_or_path,
                local_files_only=self.local_files_only,
                device=device,
            )

        classifier = _move_runtime(
            self._runtime("classifier", classifier_factory), device
        )
        target_indices = classifier.classifier_indices(artifacts.artist_keys)
        for name in SCENARIO_NAMES:
            logits = classifier.logits(
                artifacts.generated_audio[name],
                sample_rate=artifacts.sample_rate,
                batch_size=self.metric_batch_size,
            )
            stats = classifier_target_statistics(logits, target_indices)
            values[name]["target_attribution_rate"] = stats.attribution
            values[name]["target_artist_confidence"] = stats.confidence
            values[name]["target_artist_rank"] = stats.rank
        _release_runtime(classifier)

        def mert_factory() -> Any:
            from src.evaluation.suppression_similarity import MERTEncoder

            return MERTEncoder(
                self.mert_model_name_or_path,
                device=device,
                local_files_only=self.local_files_only,
            )

        mert = _move_runtime(self._runtime("mert", mert_factory), device)
        reference_embeddings = mert.encode_audio(
            artifacts.reference_audio,
            sample_rate=artifacts.sample_rate,
            batch_size=self.metric_batch_size,
        )
        for name in SCENARIO_NAMES:
            embeddings = mert.encode_audio(
                artifacts.generated_audio[name],
                sample_rate=artifacts.sample_rate,
                batch_size=self.metric_batch_size,
            )
            values[name]["mert_similarity"] = cosine_similarity_rows(
                embeddings, reference_embeddings
            )
        _release_runtime(mert)

        def clap_factory() -> Any:
            from src.evaluation.suppression_similarity import HFCLAPEncoder

            return HFCLAPEncoder(
                self.clap_model_name_or_path,
                device=device,
                local_files_only=self.local_files_only,
            )

        clap = _move_runtime(self._runtime("clap", clap_factory), device)
        for name in SCENARIO_NAMES:
            values[name]["clap_similarity"] = (
                clap.score(
                    artifacts.generated_audio[name],
                    artifacts.texts,
                    sample_rate=artifacts.sample_rate,
                )
                .detach()
                .float()
                .cpu()
            )
        _release_runtime(clap)

        def passt_factory() -> Any:
            from src.metric.quality import PaSSTKLDivergence

            return PaSSTKLDivergence()

        passt = _move_runtime(self._runtime("passt", passt_factory), device)
        for name in SCENARIO_NAMES:
            values[name]["passt_kl"] = _passt_per_sample(
                passt,
                artifacts.generated_audio[name],
                artifacts.reference_audio,
                sample_rate=artifacts.sample_rate,
                batch_size=self.metric_batch_size,
            )
        _release_runtime(passt)

        def vggish_factory() -> Any:
            from src.metric.quality import VGGishAudioEmbedding

            return VGGishAudioEmbedding()

        vggish = _move_runtime(self._runtime("vggish", vggish_factory), device)
        unique_artists = tuple(dict.fromkeys(artifacts.artist_keys))
        reference_by_artist: dict[str, Tensor] = {}
        for artist_key in unique_artists:
            indices = torch.tensor(
                [i for i, key in enumerate(artifacts.artist_keys) if key == artist_key]
            )
            reference_by_artist[artist_key] = _vggish_embeddings(
                vggish,
                artifacts.reference_audio.index_select(0, indices),
                sample_rate=artifacts.sample_rate,
            )
        for name in SCENARIO_NAMES:
            fad_values: list[Tensor] = []
            for artist_key in unique_artists:
                indices = torch.tensor(
                    [
                        i
                        for i, key in enumerate(artifacts.artist_keys)
                        if key == artist_key
                    ]
                )
                generated_embeddings = _vggish_embeddings(
                    vggish,
                    artifacts.generated_audio[name].index_select(0, indices),
                    sample_rate=artifacts.sample_rate,
                )
                fad_values.append(
                    frechet_distance_from_embeddings(
                        generated_embeddings, reference_by_artist[artist_key]
                    )
                )
            values[name]["fad"] = torch.stack(fad_values).cpu()
        _release_runtime(vggish)
        return values

    def _run_and_log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> None:
        experiment = self._wandb_experiment(trainer)
        if experiment is None:
            log.warning("Epoch control evaluation skipped: WandbLogger not found")
            self._logged_epoch = epoch
            return
        if (
            self.evaluation_runner is None
            and "classifier" not in self.runtime_factories
            and not self.classifier_checkpoint_path
        ):
            raise RuntimeError(
                "classifier_checkpoint_path is required for epoch control evaluation; "
                "set it in the callback config or disable the callback"
            )

        artifacts = self._collect_artifacts(trainer, pl_module, epoch)
        if self.evaluation_runner is None:
            values = self._evaluate_default(
                artifacts, device=self._metric_runtime_device(pl_module)
            )
        else:
            values = self.evaluation_runner(artifacts)
        summaries = summarize_scenario_metrics(values)
        rows = formatted_table_rows(summaries)
        if len(rows) != 6 or any(len(row) != 9 for row in rows):
            raise RuntimeError("control evaluation table must have 6x8 metric cells")

        import wandb

        table = wandb.Table(
            columns=["Scenario", *(METRIC_LABELS[name] for name in EVALUATION_METRICS)]
        )
        for row in rows:
            table.add_data(*row)
        payload: dict[str, Any] = {
            self.log_key: table,
            "trainer/global_step": trainer.global_step,
            f"{self.numeric_namespace}/epoch": epoch,
        }
        for scenario in SCENARIO_NAMES:
            for metric in EVALUATION_METRICS:
                summary = summaries[scenario][metric]
                prefix = f"{self.numeric_namespace}/{scenario}/{metric}"
                payload[f"{prefix}/mean"] = summary.mean
                payload[f"{prefix}/std"] = summary.std
        experiment.log(payload)
        self._logged_epoch = epoch


__all__ = ["EpochControlArtifacts", "EpochControlEvaluationCallback"]
