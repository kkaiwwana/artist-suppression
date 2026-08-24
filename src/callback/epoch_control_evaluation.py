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

from src.experiment_logging import ExperimentTracker, experiment_tracker
from src.evaluation._external_output import quiet_external_output
from src.evaluation.control_scenarios import (
    EVALUATION_METRICS,
    METRIC_LABELS,
    SCENARIO_LABELS,
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
from src.evaluation.scenario_statistics import evaluate_scenario_statistics
from src.metric.forgetting import (
    ground_truth_next_token_confidence_from_rollout_scores,
)


log = logging.getLogger(__name__)


_GT_TOKEN_CURVE_COLORS = {
    "no_control": "#4C78A8",
    "enhance_target": "#F58518",
    "suppress_single_target": "#E45756",
    "suppress_multiple_target": "#B279A2",
    "suppress_single_others": "#54A24B",
    "suppress_multiple_others": "#72B7B2",
}


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
    # the paired observations needed by the artist-level statistical tests.
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
    """Offload a cached runtime to CPU between heavyweight metric passes."""

    # Lightning validation commonly encloses callbacks in inference_mode.
    # Module.to() is allowed there, but a device copy performed in that scope
    # can manufacture inference-tensor parameters which fail when the cached
    # runtime is reused by a later callback outside that exact scope.
    with torch.inference_mode(False):
        if callable(getattr(runtime, "to", None)):
            runtime.to("cpu")
        else:
            model = getattr(runtime, "model", None)
            if model is not None and callable(getattr(model, "to", None)):
                model.to("cpu")
            if hasattr(runtime, "device"):
                runtime.device = torch.device("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_runtime(runtime: Any, device: torch.device) -> Any:
    """Move a metric runtime without requiring every wrapper to define to()."""

    with torch.inference_mode(False):
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
    generation_mode: str = "continuation"
    gt_token_confidence_by_position: Mapping[str, Tensor] | None = None
    gt_token_confidence_positions: Tensor | None = None
    teacher_forced_intervention_energy: Mapping[str, Tensor] | None = None
    prompt_audio: Tensor | None = None
    artist_index_to_key: Mapping[int, str] | None = None


class EpochControlEvaluationCallback(pl.Callback):
    """Evaluate six matched controls on a fixed balanced training cohort.

    Evaluation is queued after validation and flushed at the next train epoch
    boundary. The final pending epoch is flushed at train end, while the strategy,
    devices, and logger are still active. The cohort and multi-artist sets stay
    fixed for the full run.
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
        generation_mode: str = "continuation",
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
        log_key: str = "Control Evaluation/Results",
        numeric_namespace: str = "Control Evaluation Details",
        cache_external_models: bool = True,
        quiet_external_models: bool = True,
        hide_detailed_metrics: bool = True,
        qualitative_enabled: bool = True,
        qualitative_log_key: str = "Qualitative Comparison/Matched Six Scenarios",
        qualitative_sample_index: int = 0,
        qualitative_include_prompt: bool = True,
        gt_token_curve_max_positions: int = 100,
        gt_token_curve_interval: int = 5,
        gt_token_curve_table_log_key: str = (
            "Control Evaluation/Rollout GT Token Confidence by Position"
        ),
        gt_token_curve_plot_log_key: str = (
            "Control Evaluation/Rollout GT Token Confidence Curves by Epoch"
        ),
        gt_token_curve_y_padding_fraction: float = 0.12,
        gt_token_curve_min_y_span: float = 0.02,
        statistical_tests_enabled: bool = True,
        statistical_omnibus_log_key: str = (
            "Statistical Analysis/Scenario Omnibus Tests"
        ),
        statistical_posthoc_log_key: str = (
            "Statistical Analysis/Paired Post-hoc Tests"
        ),
        statistical_num_permutations: int = 10_000,
        statistical_num_bootstrap_resamples: int = 2_000,
        statistical_confidence: float = 0.95,
        statistical_alpha: float = 0.05,
        statistical_exact_max_artists: int = 16,
        statistical_seed: int | None = None,
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
            "gt_token_curve_max_positions": gt_token_curve_max_positions,
            "gt_token_curve_interval": gt_token_curve_interval,
            "statistical_num_permutations": statistical_num_permutations,
            "statistical_num_bootstrap_resamples": (
                statistical_num_bootstrap_resamples
            ),
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"positive values required for: {', '.join(invalid)}")
        generation_mode = str(generation_mode).strip().lower()
        if generation_mode not in {"text", "continuation"}:
            raise ValueError("generation_mode must be 'text' or 'continuation'")
        if continuation_seconds <= 0:
            raise ValueError("continuation_seconds must be positive")
        if audio_prompt_seconds < 0 or (
            generation_mode == "continuation" and audio_prompt_seconds <= 0
        ):
            raise ValueError(
                "audio_prompt_seconds must be positive in continuation mode and "
                "non-negative in text mode"
            )
        if multi_min_artists < 2 or multi_max_artists < multi_min_artists:
            raise ValueError("multi bounds must satisfy 2 <= min <= max")
        if int(qualitative_sample_index) < 0:
            raise ValueError("qualitative_sample_index must be non-negative")
        if float(gt_token_curve_y_padding_fraction) < 0:
            raise ValueError("gt_token_curve_y_padding_fraction must be non-negative")
        if not 0 < float(gt_token_curve_min_y_span) <= 1:
            raise ValueError("gt_token_curve_min_y_span must be in (0, 1]")
        if not 0 < float(statistical_confidence) < 1:
            raise ValueError("statistical_confidence must be in (0, 1)")
        if not 0 < float(statistical_alpha) < 1:
            raise ValueError("statistical_alpha must be in (0, 1)")
        if int(statistical_exact_max_artists) < 0:
            raise ValueError("statistical_exact_max_artists must be non-negative")

        self.every_n_epochs = int(every_n_epochs)
        self.num_artists = int(num_artists)
        self.clips_per_artist = int(clips_per_artist)
        self.generation_batch_size = int(generation_batch_size)
        self.metric_batch_size = int(metric_batch_size)
        self.generation_seed = int(generation_seed)
        self.cohort_seed = int(cohort_seed)
        self.generation_mode = generation_mode
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
        self.cache_external_models = bool(cache_external_models)
        self.quiet_external_models = bool(quiet_external_models)
        self.hide_detailed_metrics = bool(hide_detailed_metrics)
        self.qualitative_enabled = bool(qualitative_enabled)
        self.qualitative_log_key = str(qualitative_log_key)
        self.qualitative_sample_index = int(qualitative_sample_index)
        self.qualitative_include_prompt = bool(qualitative_include_prompt)
        self.gt_token_curve_max_positions = int(gt_token_curve_max_positions)
        self.gt_token_curve_interval = int(gt_token_curve_interval)
        self.gt_token_curve_table_log_key = str(gt_token_curve_table_log_key)
        self.gt_token_curve_plot_log_key = str(gt_token_curve_plot_log_key)
        self.gt_token_curve_y_padding_fraction = float(
            gt_token_curve_y_padding_fraction
        )
        self.gt_token_curve_min_y_span = float(gt_token_curve_min_y_span)
        self.statistical_tests_enabled = bool(statistical_tests_enabled)
        self.statistical_omnibus_log_key = str(statistical_omnibus_log_key)
        self.statistical_posthoc_log_key = str(statistical_posthoc_log_key)
        self.statistical_num_permutations = int(statistical_num_permutations)
        self.statistical_num_bootstrap_resamples = int(
            statistical_num_bootstrap_resamples
        )
        self.statistical_confidence = float(statistical_confidence)
        self.statistical_alpha = float(statistical_alpha)
        self.statistical_exact_max_artists = int(statistical_exact_max_artists)
        self.statistical_seed = int(
            generation_seed if statistical_seed is None else statistical_seed
        )
        self.fail_on_error = bool(fail_on_error)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.runtime_factories = dict(runtime_factories or {})
        self.evaluation_runner = evaluation_runner

        self._cohort: tuple[CohortItem, ...] | None = None
        self._scenario_plan: tuple[ScenarioCondition, ...] | None = None
        self._pending_epoch: int | None = None
        self._logged_epoch: int | None = None
        self._runtime_cache: dict[str, Any] = {}
        self._classifier_artist_keys: set[str] | None = None
        self._classifier_vocabulary_resolved = False
        self._backend_metrics_defined = False

    @staticmethod
    def _experiment_tracker(trainer: pl.Trainer) -> ExperimentTracker | None:
        return experiment_tracker(trainer)

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

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Flush the final epoch before Lightning tears down strategy and logging."""

        try:
            self._flush_pending(trainer, pl_module)
        finally:
            self._clear_runtime_cache()

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
        log.info("Starting control evaluation for epoch %d", epoch)
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
            log.info("Finished control evaluation for epoch %d", epoch)
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

        if self._classifier_vocabulary_resolved:
            return self._classifier_artist_keys
        if self.evaluation_runner is not None or "classifier" in self.runtime_factories:
            self._classifier_vocabulary_resolved = True
            return None
        if not self.classifier_checkpoint_path:
            self._classifier_vocabulary_resolved = True
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
        self._classifier_artist_keys = artist_keys
        self._classifier_vocabulary_resolved = True
        return self._classifier_artist_keys

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

    @staticmethod
    def _gt_ntc_max_abs_deltas(confidences: Mapping[str, Tensor]) -> dict[str, float]:
        """Compare every controlled rollout score with No Control."""

        baseline = torch.as_tensor(confidences["no_control"]).double().flatten()
        deltas: dict[str, float] = {}
        for name in SCENARIO_NAMES:
            current = torch.as_tensor(confidences[name]).double().flatten()
            if current.shape != baseline.shape:
                raise ValueError(
                    "all GT-NTC scenario vectors must have matching shapes"
                )
            finite = torch.isfinite(current) & torch.isfinite(baseline)
            deltas[name] = (
                float((current[finite] - baseline[finite]).abs().max().item())
                if bool(finite.any())
                else float("nan")
            )
        return deltas

    @classmethod
    def _diagnose_gt_ntc(
        cls,
        confidences: Mapping[str, Tensor],
        intervention_energies: Mapping[str, Sequence[float]],
    ) -> None:
        """Warn when all controlled results are truly identical, not rounded."""

        deltas = cls._gt_ntc_max_abs_deltas(confidences)
        controlled_deltas = [
            value for name, value in deltas.items() if name != "no_control"
        ]
        if not controlled_deltas or not all(
            value == 0.0 for value in controlled_deltas
        ):
            return
        observed_energies = [
            abs(float(value))
            for name, values in intervention_energies.items()
            if name != "no_control"
            for value in values
            if math.isfinite(float(value))
        ]
        max_energy = max(observed_energies, default=0.0)
        if max_energy == 0.0:
            log.warning(
                "Rollout GT-NTC is exactly identical for all six scenarios and the "
                "controlled teacher-forcing intervention energy is zero. This "
                "usually means the learned control residual is still zero (or "
                "intervention_scale is zero), rather than a statistics mix-up."
            )
        else:
            log.warning(
                "Rollout GT-NTC is exactly identical for all six scenarios even "
                "though "
                "controlled teacher-forcing intervention energy is non-zero "
                "(max=%g). Inspect concept hooks and logits for this checkpoint.",
                max_energy,
            )

    def _collect_artifacts(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> EpochControlArtifacts:
        dataset, records, artist_to_index, collator = self._training_data(trainer)
        device = torch.device(getattr(pl_module, "device", "cpu"))
        frame_rate = float(audio_frame_rate(pl_module))
        use_audio_prompt = self.generation_mode == "continuation"
        prompt_frames = (
            max(1, int(round(self.audio_prompt_seconds * frame_rate)))
            if use_audio_prompt
            else 0
        )
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

        curve_max_positions = min(
            self.gt_token_curve_max_positions,
            continuation_frames,
        )
        curve_positions = tuple(
            dict.fromkeys(
                (
                    1,
                    *range(
                        self.gt_token_curve_interval,
                        curve_max_positions + 1,
                        self.gt_token_curve_interval,
                    ),
                )
            )
        )
        generated: dict[str, list[Tensor]] = {name: [] for name in SCENARIO_NAMES}
        confidences: dict[str, list[Tensor]] = {name: [] for name in SCENARIO_NAMES}
        confidence_curves: dict[str, list[Tensor]] = {
            name: [] for name in SCENARIO_NAMES
        }
        intervention_energies: dict[str, list[float]] = {
            name: [] for name in SCENARIO_NAMES
        }
        prompts: list[Tensor] = []
        references: list[Tensor] = []
        texts: list[str] = []
        artist_keys: list[str] = []
        artist_ids: list[Tensor] = []
        decode = getattr(getattr(pl_module, "model", None), "decode_audio_tokens", None)
        run_generator = getattr(pl_module, "_run_generator", None)
        generate_output = getattr(
            pl_module, "_generate_with_condition_output", None
        )
        if not all(
            callable(method) for method in (decode, run_generator, generate_output)
        ):
            raise RuntimeError(
                "model must expose decode_audio_tokens(), _run_generator(), and "
                "_generate_with_condition_output()"
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
            if use_audio_prompt:
                prompts.append(
                    fixed_audio_segment(
                        reference_audio,
                        sample_rate=sample_rate,
                        start_seconds=0.0,
                        duration_seconds=encoded_prompt_seconds,
                    )
                    .detach()
                    .float()
                    .cpu()
                )
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
            generation_batch = (
                attach_audio_prompt(batch, prompt_frames=prompt_frames)
                if use_audio_prompt
                else batch
            )
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
                labels = teacher_forced["labels"]
                pad_token_id = int(
                    getattr(
                        getattr(pl_module, "model", None), "audio_pad_token_id", 2048
                    )
                )
                intervention_energy = teacher_forced.get("intervention_energy")
                if (
                    isinstance(intervention_energy, Tensor)
                    and intervention_energy.numel()
                ):
                    intervention_energies[scenario.name].append(
                        float(intervention_energy.detach().float().max().cpu().item())
                    )
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(batch_seed)
                    generation_output = generate_output(
                        generation_batch, generation_kwargs, condition
                    )
                generated_audio = generation_output.audio_values
                rollout_scores = getattr(generation_output.raw_output, "scores", None)
                if rollout_scores is None:
                    raise RuntimeError(
                        "MusicGen output_scores are required for rollout GT-NTC"
                    )
                sampled_curve = (
                    ground_truth_next_token_confidence_from_rollout_scores(
                        rollout_scores,
                        labels,
                        prompt_frames=prompt_frames,
                        positions=curve_positions,
                        pad_token_id=pad_token_id,
                    )
                )
                confidence_curves[scenario.name].append(sampled_curve.cpu())
                finite_curve = torch.isfinite(sampled_curve)
                curve_count = finite_curve.sum(dim=1)
                gt_confidence = (
                    sampled_curve.masked_fill(~finite_curve, 0.0).sum(dim=1)
                    / curve_count.clamp_min(1)
                ).masked_fill(curve_count.eq(0), float("nan"))
                confidences[scenario.name].append(gt_confidence.cpu())
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

        combined_confidences = {
            name: torch.cat(confidences[name]) for name in SCENARIO_NAMES
        }
        self._diagnose_gt_ntc(combined_confidences, intervention_energies)
        combined_intervention_energies = {
            name: torch.tensor(intervention_energies[name], dtype=torch.float32)
            for name in SCENARIO_NAMES
        }
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
            gt_token_confidence=combined_confidences,
            generation_mode=self.generation_mode,
            gt_token_confidence_by_position={
                name: torch.cat(confidence_curves[name]) for name in SCENARIO_NAMES
            },
            gt_token_confidence_positions=torch.tensor(
                curve_positions, dtype=torch.long
            ),
            teacher_forced_intervention_energy=combined_intervention_energies,
            prompt_audio=torch.cat(prompts) if prompts else None,
            artist_index_to_key={
                int(index): str(key) for key, index in artist_to_index.items()
            },
        )

    def _runtime(self, name: str, default_factory: Callable[[], Any]) -> Any:
        if self.cache_external_models and name in self._runtime_cache:
            return self._runtime_cache[name]
        factory = self.runtime_factories.get(name, default_factory)
        log.info("Loading external evaluation runtime '%s' (cached after load)", name)
        with quiet_external_output(self.quiet_external_models):
            # Model parameters created under inference_mode are special
            # inference tensors. They cannot safely survive the CPU/GPU
            # offload-and-reuse cycle used by this cache.
            with torch.inference_mode(False):
                runtime = factory()
        if self.cache_external_models:
            self._runtime_cache[name] = runtime
        return runtime

    def _clear_runtime_cache(self) -> None:
        for runtime in self._runtime_cache.values():
            _release_runtime(runtime)
        self._runtime_cache.clear()

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

            return PaSSTKLDivergence(quiet_backend=self.quiet_external_models)

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

            return VGGishAudioEmbedding(quiet_backend=self.quiet_external_models)

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

    def _define_backend_metrics(self, tracker: ExperimentTracker) -> None:
        """Hide detailed W&B panels; SwanLab needs no equivalent setup."""

        if self._backend_metrics_defined:
            return
        if self.hide_detailed_metrics:
            hidden_patterns = (
                f"{self.numeric_namespace}/*",
                # Hide panels created by runs using the previous callback keys.
                "evaluation/control_metric_stats/*",
                "evaluation/control_metrics",
            )
            for pattern in dict.fromkeys(hidden_patterns):
                tracker.define_metric(pattern, hidden=True)
        self._backend_metrics_defined = True

    def _qualitative_comparison_table(
        self,
        media_module: Any,
        artifacts: EpochControlArtifacts,
    ) -> Any | None:
        """Build one matched row containing the six already-generated audios."""

        if not self.qualitative_enabled:
            return None
        sample_count = len(artifacts.artist_keys)
        if sample_count == 0:
            raise ValueError("qualitative comparison needs at least one cohort sample")
        index = self.qualitative_sample_index
        if index >= sample_count:
            raise ValueError(
                "qualitative_sample_index is outside the fixed evaluation cohort: "
                f"{index} >= {sample_count}"
            )
        if self._scenario_plan is None:
            raise RuntimeError("qualitative comparison requires a scenario plan")
        scenario_plan = {scenario.name: scenario for scenario in self._scenario_plan}
        artist_names = artifacts.artist_index_to_key or {}
        prompt = artifacts.prompt_audio
        prefix_seconds = 0.0
        include_prompt = self.qualitative_include_prompt and prompt is not None
        if (
            self.qualitative_include_prompt
            and artifacts.generation_mode == "continuation"
        ):
            if not include_prompt:
                raise RuntimeError(
                    "qualitative_include_prompt requires prompt_audio artifacts"
                )
        if include_prompt:
            prefix_seconds = float(prompt.shape[-1]) / artifacts.sample_rate

        audio_cells: list[Any] = []
        for name in SCENARIO_NAMES:
            tail = artifacts.generated_audio[name][index].detach().float().cpu()
            if tail.ndim == 2:
                tail = tail[0]
            if tail.ndim != 1:
                raise ValueError("qualitative generated audio must be mono or [C,T]")
            waveform = tail
            if include_prompt:
                shared_prefix = prompt[index].detach().float().cpu()
                if shared_prefix.ndim == 2:
                    shared_prefix = shared_prefix[0]
                if shared_prefix.ndim != 1:
                    raise ValueError("qualitative prompt audio must be mono or [C,T]")
                waveform = torch.cat((shared_prefix, tail), dim=-1)

            artist_ids = scenario_plan[name].artist_sets[index]
            controlled_artists = [
                artist_names.get(int(artist_id), f"artist_id:{int(artist_id)}")
                for artist_id in artist_ids
            ]
            control_text = ", ".join(controlled_artists) or "none"
            audio_cells.append(
                media_module.Audio(
                    waveform.numpy(),
                    sample_rate=artifacts.sample_rate,
                    caption=(
                        f"{SCENARIO_LABELS[name]}; controlled artists: "
                        f"{control_text}"
                    ),
                )
            )

        table = media_module.Table(
            columns=[
                "epoch",
                "generation_mode",
                "cohort_sample",
                "target_artist",
                "caption",
                "shared_real_prefix_seconds",
                *(SCENARIO_LABELS[name] for name in SCENARIO_NAMES),
            ]
        )
        table.add_data(
            artifacts.epoch,
            artifacts.generation_mode,
            index,
            artifacts.artist_keys[index],
            artifacts.texts[index],
            prefix_seconds,
            *audio_cells,
        )
        return table

    def _gt_token_curve_payload(
        self,
        media_module: Any,
        artifacts: EpochControlArtifacts,
    ) -> dict[str, Any]:
        """Build a raw table and fixed-color image for the tracker history."""

        curves = artifacts.gt_token_confidence_by_position
        positions = artifacts.gt_token_confidence_positions
        if curves is None or positions is None:
            return {}
        positions = torch.as_tensor(positions).long().flatten()
        if positions.numel() == 0:
            return {}

        means: dict[str, Tensor] = {}
        for scenario in SCENARIO_NAMES:
            if scenario not in curves:
                raise KeyError(f"GT-NTC curve is missing scenario {scenario!r}")
            values = torch.as_tensor(curves[scenario]).double()
            if values.ndim != 2 or values.shape[1] != positions.numel():
                raise ValueError(
                    "GT-NTC curves must have shape [clips, positions]; "
                    f"got {tuple(values.shape)} for {positions.numel()} positions"
                )
            finite = torch.isfinite(values)
            count = finite.sum(dim=0)
            total = values.masked_fill(~finite, 0.0).sum(dim=0)
            mean = total / count.clamp_min(1)
            means[scenario] = mean.masked_fill(count.eq(0), float("nan"))

        all_means = torch.cat([means[name] for name in SCENARIO_NAMES])
        finite_means = all_means[torch.isfinite(all_means)]
        if finite_means.numel():
            data_min = float(finite_means.min().item())
            data_max = float(finite_means.max().item())
            data_span = data_max - data_min
            y_span = max(
                self.gt_token_curve_min_y_span,
                data_span * (1.0 + 2.0 * self.gt_token_curve_y_padding_fraction),
            )
            y_span = min(1.0, y_span)
            y_center = 0.5 * (data_min + data_max)
            y_lower = y_center - 0.5 * y_span
            y_upper = y_center + 0.5 * y_span
            if y_lower < 0.0:
                y_upper -= y_lower
                y_lower = 0.0
            if y_upper > 1.0:
                y_lower -= y_upper - 1.0
                y_upper = 1.0
            y_lower = max(0.0, y_lower)
            y_upper = min(1.0, y_upper)
        else:
            y_lower, y_upper = 0.0, 1.0

        table = media_module.Table(
            columns=[
                "Rollout Token Position",
                *(SCENARIO_LABELS[name] for name in SCENARIO_NAMES),
            ]
        )
        for column, position in enumerate(positions.tolist()):
            table.add_data(
                int(position),
                *(float(means[name][column].item()) for name in SCENARIO_NAMES),
            )

        payload: dict[str, Any] = {self.gt_token_curve_table_log_key: table}
        image_type = getattr(media_module, "Image", None)
        if callable(image_type):
            # Construct the figure directly instead of through pyplot so epoch-end
            # evaluation does not accumulate figures in matplotlib's global state.
            # Some media adapters expect pyplot to be initialized while they
            # type-check a matplotlib Figure.
            from matplotlib import pyplot as _matplotlib_pyplot  # noqa: F401
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from matplotlib.ticker import FormatStrFormatter

            figure = Figure(figsize=(10.5, 6.2), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.add_subplot(1, 1, 1)
            x_values = positions.tolist()
            for scenario in SCENARIO_NAMES:
                axes.plot(
                    x_values,
                    means[scenario].tolist(),
                    color=_GT_TOKEN_CURVE_COLORS[scenario],
                    linestyle="-",
                    linewidth=2.2,
                    label=SCENARIO_LABELS[scenario],
                )
            axes.set(
                title=(
                    "GT Token Confidence during Free Rollout "
                    f"({artifacts.generation_mode}, Epoch {artifacts.epoch}, "
                    "zoomed y-axis)"
                ),
                xlabel="Rollout Token Position",
                ylabel="Mean GT Token Confidence",
                ylim=(y_lower, y_upper),
            )
            axes.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
            axes.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.7)
            axes.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=3,
                frameon=False,
            )
            payload[self.gt_token_curve_plot_log_key] = image_type(
                figure,
                caption=(
                    f"Epoch {artifacts.epoch}: fixed colors identify the six "
                    "control scenarios; zoomed y-axis "
                    f"[{y_lower:.4f}, {y_upper:.4f}]."
                ),
            )
        return payload

    def _statistical_table_payload(
        self,
        media_module: Any,
        values: Mapping[str, Mapping[str, Any]],
        artifacts: EpochControlArtifacts,
    ) -> dict[str, Any]:
        """Build two fixed tables without creating per-test scalar panels."""

        if not self.statistical_tests_enabled:
            return {}
        omnibus_results, pairwise_results = evaluate_scenario_statistics(
            values,
            metrics=EVALUATION_METRICS,
            scenarios=SCENARIO_NAMES,
            artist_keys=artifacts.artist_keys,
            num_permutations=self.statistical_num_permutations,
            num_bootstrap_resamples=self.statistical_num_bootstrap_resamples,
            confidence=self.statistical_confidence,
            alpha=self.statistical_alpha,
            exact_max_pairs=self.statistical_exact_max_artists,
            seed=self.statistical_seed + int(artifacts.epoch),
        )

        omnibus_table = media_module.Table(
            columns=[
                "Metric",
                "Unit",
                "N Artists",
                "RM-ANOVA F",
                "df1",
                "df2",
                "RM-ANOVA p",
                "GG Epsilon",
                "GG df1",
                "GG df2",
                "GG-corrected p",
                "Friedman Chi-square",
                "Friedman p",
                f"Significant (GG p < {self.statistical_alpha:g})",
            ]
        )
        for result in omnibus_results:
            omnibus_table.add_data(
                METRIC_LABELS[result.metric],
                "artist",
                result.count,
                result.f_statistic,
                result.df_numerator,
                result.df_denominator,
                result.p_value,
                result.gg_epsilon,
                result.gg_df_numerator,
                result.gg_df_denominator,
                result.gg_p_value,
                result.friedman_statistic,
                result.friedman_p_value,
                bool(result.gg_p_value < self.statistical_alpha),
            )

        confidence_label = f"{100 * self.statistical_confidence:g}% CI"
        posthoc_table = media_module.Table(
            columns=[
                "Metric",
                "Scenario A",
                "Scenario B",
                "Unit",
                "N Artists",
                "Mean A",
                "Mean B",
                "Mean Delta (B - A)",
                f"{confidence_label} Low",
                f"{confidence_label} High",
                "Paired Mean Permutation p",
                "Holm p (15 comparisons)",
                "Significant after Omnibus + Holm",
            ]
        )
        for result in pairwise_results:
            posthoc_table.add_data(
                METRIC_LABELS[result.metric],
                SCENARIO_LABELS[result.scenario_a],
                SCENARIO_LABELS[result.scenario_b],
                "artist",
                result.count,
                result.mean_a,
                result.mean_b,
                result.mean_difference,
                result.ci_low,
                result.ci_high,
                result.permutation_p_value,
                result.holm_p_value,
                result.significant,
            )
        return {
            self.statistical_omnibus_log_key: omnibus_table,
            self.statistical_posthoc_log_key: posthoc_table,
        }

    def _run_and_log(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> None:
        tracker = self._experiment_tracker(trainer)
        if tracker is None:
            log.warning(
                "Epoch control evaluation skipped: supported experiment "
                "logger not found"
            )
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
        expected_width = 1 + len(EVALUATION_METRICS)
        if len(rows) != len(SCENARIO_NAMES) or any(
            len(row) != expected_width for row in rows
        ):
            raise RuntimeError("control evaluation table has an invalid shape")

        media = tracker.media
        self._define_backend_metrics(tracker)
        table = media.Table(
            columns=["Scenario", *(METRIC_LABELS[name] for name in EVALUATION_METRICS)]
        )
        for row in rows:
            table.add_data(*row)
        payload: dict[str, Any] = {
            self.log_key: table,
            "trainer/global_step": trainer.global_step,
            f"{self.numeric_namespace}/epoch": epoch,
        }
        qualitative_table = self._qualitative_comparison_table(media, artifacts)
        if qualitative_table is not None:
            payload[self.qualitative_log_key] = qualitative_table
        payload.update(self._gt_token_curve_payload(media, artifacts))
        payload.update(self._statistical_table_payload(media, values, artifacts))
        for scenario in SCENARIO_NAMES:
            for metric in EVALUATION_METRICS:
                summary = summaries[scenario][metric]
                prefix = f"{self.numeric_namespace}/{scenario}/{metric}"
                payload[f"{prefix}/mean"] = summary.mean
        for scenario, delta in self._gt_ntc_max_abs_deltas(
            artifacts.gt_token_confidence
        ).items():
            payload[
                f"{self.numeric_namespace}/diagnostics/gt_ntc/"
                f"{scenario}/max_abs_delta_vs_no_control"
            ] = delta
        if artifacts.teacher_forced_intervention_energy is not None:
            for (
                scenario,
                energy,
            ) in artifacts.teacher_forced_intervention_energy.items():
                energy = torch.as_tensor(energy).float().flatten()
                if energy.numel():
                    prefix = (
                        f"{self.numeric_namespace}/diagnostics/gt_ntc/" f"{scenario}"
                    )
                    payload[f"{prefix}/intervention_energy_mean"] = float(
                        energy.mean().item()
                    )
                    payload[f"{prefix}/intervention_energy_max"] = float(
                        energy.max().item()
                    )
        tracker.log(payload)
        self._logged_epoch = epoch


__all__ = ["EpochControlArtifacts", "EpochControlEvaluationCallback"]
