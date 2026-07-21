"""Lightning training for explicit opt-in artist suppression controls."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model.base import BaseGenerationModel, _cfg_get, _instantiate_component
from src.model.module.concept_learner import ConceptCondition


class UnlearnableGenerationModel(BaseGenerationModel):
    """Learn artist directions without an automatic router.

    Normal inference uses the untouched generator with no hooks. During
    training, four explicit paths give each learned direction a useful and
    bounded meaning:

    * default: no control, used as the stable reference;
    * positive: add artist A and improve A's teacher-forced likelihood;
    * negative: subtract A and mirror the positive likelihood change;
    * preservation: subtract A while modeling a different artist B.

    The positive path stops at finite gain and wrong-artist margins. The
    negative path is never trained with unbounded reverse cross entropy. Its
    effect is matched to that finite improvement, while the B path prevents a
    generic loss of musical ability. A relative residual-energy term and hard
    block-scale bound limit the actual intervention received by MusicGen.
    """

    CONTROL_BATCH_KEYS = {
        "concept_ids",
        "concept_targets",
        "concept_weights",
        "control_direction",
        "suppressed_artist_ids",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.concept_learner = _instantiate_component(
            _cfg_get(self.model_cfg, "concept_learner")
        )
        self.objective_cfg = _cfg_get(self.model_cfg, "control_training", {})
        self.loss_cfg = _cfg_get(self.model_cfg, "losses", {})
        self.hook_blocks = _cfg_get(self.model_cfg, "hook_blocks")
        self.intervention_scale = _cfg_get(
            self.model_cfg,
            "intervention_scale",
            1.0,
        )
        self.validation_full_precision = bool(
            _cfg_get(self.model_cfg, "validation_full_precision", False)
        )
        self.train_generator = bool(
            _cfg_get(self.model_cfg, "train_generator", False)
        )
        if not self.train_generator:
            self.model.requires_grad_(False)
            self.model.eval()
        self._validate_objective_config()

    def train(self, mode: bool = True):
        """Keep the frozen MusicGen reference deterministic during training."""

        super().train(mode)
        if not self.train_generator:
            self.model.eval()
        return self

    def _validate_objective_config(self) -> None:
        values = {
            "default_weight": _cfg_get(self.objective_cfg, "default_weight", 0.0),
            "positive_weight": _cfg_get(self.objective_cfg, "positive_weight", 0.0),
            "positive_margin_weight": _cfg_get(
                self.objective_cfg, "positive_margin_weight", 0.0
            ),
            "suppression_symmetry_weight": _cfg_get(
                self.objective_cfg, "suppression_symmetry_weight", 1.0
            ),
        }
        preservation = _cfg_get(self.objective_cfg, "preservation", {})
        values["preservation.weight"] = _cfg_get(preservation, "weight", 1.0)
        contrastive = _cfg_get(self.objective_cfg, "artist_contrastive", {})
        values["artist_contrastive.weight"] = _cfg_get(
            contrastive, "weight", 0.0
        )
        values["losses.intervention_energy_weight"] = _cfg_get(
            self.loss_cfg, "intervention_energy_weight", 0.0
        )
        for name, value in values.items():
            if float(value) < 0:
                raise ValueError(f"control_training.{name} must be non-negative")
        for name, value in {
            "positive_margin": _cfg_get(self.objective_cfg, "positive_margin", 0.0),
            "preservation.margin": _cfg_get(preservation, "margin", 0.0),
            "artist_contrastive.margin": _cfg_get(contrastive, "margin", 0.0),
            "losses.intervention_energy_epsilon": _cfg_get(
                self.loss_cfg, "intervention_energy_epsilon", 1e-6
            ),
        }.items():
            if float(value) < 0:
                raise ValueError(f"control_training.{name} must be non-negative")
        if float(_cfg_get(self.loss_cfg, "intervention_energy_epsilon", 1e-6)) <= 0:
            raise ValueError("losses.intervention_energy_epsilon must be positive")
        budget = _cfg_get(
            self.loss_cfg,
            "intervention_global_rms_budget",
            0.0,
        )
        if budget is not None and float(budget) < 0:
            raise ValueError(
                "losses.intervention_global_rms_budget must be non-negative"
            )

    @staticmethod
    def _first(batch: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in batch and batch[key] is not None:
                return batch[key]
        return None

    def _artist_targets(self, batch: Mapping[str, Any]) -> Tensor:
        targets = self._first(
            batch,
            "concept_targets",
            "artist_label",
            "concept_ids",
        )
        if targets is None:
            raise ValueError(
                "explicit artist control requires concept_targets, "
                "artist_label, or concept_ids"
            )
        targets = torch.as_tensor(targets, device=self.device)
        if targets.ndim != 1:
            raise ValueError("artist targets must have shape [batch]")
        return targets.long()

    def _generator_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in batch.items()
            if key not in self.CONTROL_BATCH_KEYS
        }

    def _generation_uses_cfg(self, batch: Mapping[str, Any]) -> bool:
        generation_kwargs = batch.get("generation_kwargs")
        supplied = (
            generation_kwargs.get("guidance_scale")
            if isinstance(generation_kwargs, Mapping)
            else None
        )
        guidance_scale = (
            supplied
            if supplied is not None
            else _cfg_get(self.test_generation_cfg, "guidance_scale", 1.0)
        )
        return guidance_scale is not None and float(guidance_scale) > 1.0

    def _run_generator(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        *,
        stage: str,
        condition: Optional[ConceptCondition] = None,
    ) -> Dict[str, Any]:
        handles = []
        intervention_energies: list[Tensor] = []
        intervention_energy_by_block: Dict[int, Tensor] = {}

        def _capture_intervention_energy(
            reference_hidden: Tensor,
            details: Mapping[str, Tensor],
            context: Any,
        ) -> None:
            delta = details.get("intervention_delta")
            if not isinstance(delta, Tensor):
                raise TypeError(
                    "concept learner details must contain intervention_delta"
                )
            epsilon = float(
                _cfg_get(self.loss_cfg, "intervention_energy_epsilon", 1e-6)
            )
            # Keep the denominator a fixed reference. Otherwise the controller
            # could reduce the ratio by inflating the hidden state inherited
            # from earlier controlled blocks.
            reference_energy = (
                reference_hidden.detach().float().square().mean().clamp_min(epsilon)
            )
            relative_energy = delta.float().square().mean() / reference_energy
            intervention_energies.append(relative_energy)
            intervention_energy_by_block[int(context.block_index)] = relative_energy

        if condition is not None:
            handles = self.model.register_concept_learner_hooks(
                self.concept_learner,
                condition=condition,
                module_names=self.hook_blocks,
                intervention_scale=self.intervention_scale,
                batch_repeat_interleave=int(
                    getattr(self.model, "num_codebooks", 1) or 1
                ),
                details_callback=_capture_intervention_energy,
            )
        try:
            generator_batch = self._generator_batch(batch)
            if stage == "train":
                result = self.model.training_step(generator_batch, batch_idx)
            else:
                result = self.model.validation_step(generator_batch, batch_idx)
            result = dict(result)
            zero = result["sample_losses"].sum() * 0.0
            if intervention_energies:
                stacked = torch.stack(intervention_energies)
                # Sum squared relative RMS over controlled blocks. Averaging
                # would charge the same penalty for one perturbed block and
                # twelve equally perturbed blocks even though the latter has a
                # much larger accumulated effect during autoregressive rollout.
                result["intervention_energy"] = stacked.sum()
                result["intervention_energy_max"] = stacked.max()
            else:
                result["intervention_energy"] = zero
                result["intervention_energy_max"] = zero
            result["intervention_energy_by_block"] = intervention_energy_by_block
            return result
        finally:
            if handles:
                self.model.remove_hidden_state_hooks(handles)

    @classmethod
    def _select_rows(cls, value: Any, indices: Tensor, batch_size: int) -> Any:
        if (
            isinstance(value, Tensor)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            return value.index_select(0, indices.to(value.device))
        if isinstance(value, Mapping):
            return {
                key: cls._select_rows(item, indices, batch_size)
                for key, item in value.items()
            }
        index_list = indices.detach().cpu().tolist()
        if isinstance(value, list) and len(value) == batch_size:
            return [value[index] for index in index_list]
        if isinstance(value, tuple) and len(value) == batch_size:
            return tuple(value[index] for index in index_list)
        return value

    @classmethod
    def _select_batch_rows(
        cls,
        batch: Mapping[str, Any],
        indices: Tensor,
        batch_size: int,
    ) -> Dict[str, Any]:
        selected = {
            key: cls._select_rows(value, indices, batch_size)
            for key, value in batch.items()
        }
        selected.pop("encoder_outputs", None)
        return selected

    @staticmethod
    def _cross_artist_pairs(targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Pair each possible source A with a different donor B in the batch."""

        sources: list[int] = []
        donors: list[int] = []
        values = targets.detach().cpu().tolist()
        for source, source_target in enumerate(values):
            candidates = [
                index
                for index, target in enumerate(values)
                if target != source_target
            ]
            if not candidates:
                continue
            sources.append(source)
            donors.append(candidates[source % len(candidates)])
        device = targets.device
        return (
            torch.tensor(sources, device=device, dtype=torch.long),
            torch.tensor(donors, device=device, dtype=torch.long),
        )

    @staticmethod
    def _positive_margin_loss(
        positive_losses: Tensor,
        default_losses: Tensor,
        margin: float,
    ) -> Tensor:
        return torch.relu(positive_losses - default_losses.detach() + margin).mean()

    def _wrong_artist_targets(self, targets: Tensor, batch_idx: int) -> Optional[Tensor]:
        """Choose deterministic, always-different artist negatives.

        Negatives need not occur elsewhere in the mini-batch. Cycling the
        offset across steps exposes each artist direction to many other
        artists without relying on a large batch size.
        """

        num_concepts = int(getattr(self.concept_learner, "num_concepts", 0))
        if num_concepts < 2:
            return None
        offset = 1 + (int(batch_idx) % (num_concepts - 1))
        return (targets + offset) % num_concepts

    @staticmethod
    def _artist_contrastive_loss(
        correct_losses: Tensor,
        wrong_losses: Tensor,
        margin: float,
    ) -> Tensor:
        """Require the correct artist control to beat a wrong control.

        The wrong branch is a detached reference. Consequently this objective
        can only improve the correct condition; it never adversarially makes a
        wrong condition noisy or increases its language-model loss.
        """

        return torch.relu(
            correct_losses - wrong_losses.detach() + float(margin)
        ).mean()

    def _scale_monitors(self) -> Dict[str, Tensor]:
        effective = self.concept_learner.effective_block_scales
        monitors: Dict[str, Tensor] = {
            "block_scale_abs_mean_monitor": effective.abs().mean(),
            "block_scale_abs_max_monitor": effective.abs().max(),
            "block_scale_raw_abs_max_monitor": self.concept_learner.block_scales.abs().max(),
        }
        for block_index, scale in enumerate(effective):
            monitors[f"block_{block_index:02d}_effective_scale_monitor"] = scale
        return monitors

    @staticmethod
    def _energy_monitors(
        result: Mapping[str, Any],
        *,
        prefix: str,
    ) -> Dict[str, Tensor]:
        energy = result["intervention_energy"]
        maximum = result["intervention_energy_max"]
        monitors: Dict[str, Tensor] = {
            f"{prefix}_intervention_energy_monitor": energy,
            f"{prefix}_intervention_relative_rms_monitor": energy.clamp_min(0).sqrt(),
            f"{prefix}_intervention_relative_rms_max_monitor": maximum.clamp_min(0).sqrt(),
        }
        by_block = result.get("intervention_energy_by_block", {})
        if isinstance(by_block, Mapping):
            for block_index, block_energy in by_block.items():
                monitors[
                    f"{prefix}_block_{int(block_index):02d}_relative_rms_monitor"
                ] = block_energy.clamp_min(0).sqrt()
        return monitors

    def _intervention_budget_excess(self, energy: Tensor) -> Tensor:
        """Penalize only global intervention RMS above the configured budget.

        ``energy`` is already ``sum_l (RMS(delta_l) / RMS(h_l))**2``.
        A positive budget therefore describes one network-wide RMS allowance,
        independent of how many decoder blocks are hooked. Setting the budget
        to zero recovers a direct summed-energy penalty.
        """

        budget = _cfg_get(
            self.loss_cfg,
            "intervention_global_rms_budget",
            0.0,
        )
        if budget is None or float(budget) == 0.0:
            return energy
        epsilon = float(
            _cfg_get(self.loss_cfg, "intervention_energy_epsilon", 1e-6)
        )
        global_relative_rms = energy.clamp_min(epsilon).sqrt()
        return F.relu(global_relative_rms - float(budget)).square()

    @staticmethod
    def _suppression_symmetry_loss(
        default_losses: Tensor,
        positive_losses: Tensor,
        negative_losses: Tensor,
    ) -> Tensor:
        """Match negative degradation to the finite positive improvement."""

        positive_gain = default_losses.detach() - positive_losses.detach()
        negative_effect = negative_losses - default_losses.detach()
        return F.smooth_l1_loss(negative_effect, positive_gain)

    @staticmethod
    def _preservation_hinge(
        controlled_losses: Tensor,
        reference_losses: Tensor,
        margin: float,
    ) -> Tensor:
        return torch.relu(
            controlled_losses - reference_losses.detach() - margin
        ).mean()

    def _cross_artist_preservation(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        *,
        stage: str,
        targets: Tensor,
        default_losses: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        zero = default_losses.sum() * 0.0
        preservation = _cfg_get(self.objective_cfg, "preservation", {})
        if not bool(_cfg_get(preservation, "enabled", True)):
            return zero, zero, zero, zero, zero
        source_indices, donor_indices = self._cross_artist_pairs(targets)
        pair_rate = default_losses.new_tensor(
            source_indices.numel() / max(1, targets.numel())
        )
        if source_indices.numel() == 0:
            return zero, zero, zero, pair_rate, zero

        donor_batch = self._select_batch_rows(
            batch,
            donor_indices,
            targets.shape[0],
        )
        source_targets = targets.index_select(0, source_indices)
        condition = self.concept_learner.suppression_condition(source_targets)
        controlled = self._run_generator(
            donor_batch,
            batch_idx,
            stage=stage,
            condition=condition,
        )
        controlled_losses = controlled["sample_losses"]
        reference_losses = default_losses.index_select(0, donor_indices).detach()
        margin = float(_cfg_get(preservation, "margin", 0.0))
        preservation_loss = self._preservation_hinge(
            controlled_losses,
            reference_losses,
            margin,
        )
        return (
            controlled_losses.mean(),
            reference_losses.mean(),
            preservation_loss,
            pair_rate,
            controlled["intervention_energy"],
        )

    def _full_control_objective(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        *,
        stage: str,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        targets = self._artist_targets(batch)
        batch_size = targets.shape[0]

        default_result = self._run_generator(
            batch,
            batch_idx,
            stage=stage,
            condition=None,
        )
        positive_result = self._run_generator(
            batch,
            batch_idx,
            stage=stage,
            condition=self.concept_learner.positive_condition(targets),
        )
        negative_result = self._run_generator(
            batch,
            batch_idx,
            stage=stage,
            condition=self.concept_learner.suppression_condition(targets),
        )

        default_losses = default_result["sample_losses"]
        positive_losses = positive_result["sample_losses"]
        negative_losses = negative_result["sample_losses"]
        default_loss = default_losses.mean()
        positive_loss = positive_losses.mean()
        negative_loss = negative_losses.mean()

        positive_margin = float(
            _cfg_get(self.objective_cfg, "positive_margin", 0.0)
        )
        positive_margin_loss = self._positive_margin_loss(
            positive_losses,
            default_losses,
            positive_margin,
        )
        symmetry_loss = self._suppression_symmetry_loss(
            default_losses,
            positive_losses,
            negative_losses,
        )
        contrastive_cfg = _cfg_get(
            self.objective_cfg,
            "artist_contrastive",
            {},
        )
        wrong_targets = self._wrong_artist_targets(targets, batch_idx)
        contrastive_pair_rate = default_loss.new_tensor(
            wrong_targets is not None
        )
        wrong_positive_loss = default_loss.new_zeros(())
        artist_contrastive_loss = default_loss.new_zeros(())
        if (
            bool(_cfg_get(contrastive_cfg, "enabled", True))
            and wrong_targets is not None
        ):
            with torch.no_grad():
                wrong_result = self._run_generator(
                    batch,
                    batch_idx,
                    stage=stage,
                    condition=self.concept_learner.positive_condition(
                        wrong_targets
                    ),
                )
            wrong_losses = wrong_result["sample_losses"]
            wrong_positive_loss = wrong_losses.mean()
            artist_contrastive_loss = self._artist_contrastive_loss(
                positive_losses,
                wrong_losses,
                float(_cfg_get(contrastive_cfg, "margin", 0.0)),
            )
        (
            cross_raw,
            cross_reference,
            preservation_loss,
            pair_rate,
            cross_energy,
        ) = self._cross_artist_preservation(
            batch,
            batch_idx,
            stage=stage,
            targets=targets,
            default_losses=default_losses,
        )

        regularizers = self.concept_learner.compute_losses(
            style_bank_l2_weight=float(
                _cfg_get(self.loss_cfg, "style_bank_l2_weight", 0.0)
            ),
            concept_center_weight=float(
                _cfg_get(self.loss_cfg, "concept_center_weight", 0.0)
            ),
        )
        default_weight = float(
            _cfg_get(self.objective_cfg, "default_weight", 0.0)
        )
        positive_weight = float(
            _cfg_get(self.objective_cfg, "positive_weight", 0.0)
        )
        positive_margin_weight = float(
            _cfg_get(self.objective_cfg, "positive_margin_weight", 0.0)
        )
        symmetry_weight = float(
            _cfg_get(self.objective_cfg, "suppression_symmetry_weight", 1.0)
        )
        preservation_cfg = _cfg_get(self.objective_cfg, "preservation", {})
        preservation_weight = float(
            _cfg_get(preservation_cfg, "weight", 1.0)
        )
        contrastive_weight = float(
            _cfg_get(contrastive_cfg, "weight", 0.0)
        )
        energy_weight = float(
            _cfg_get(self.loss_cfg, "intervention_energy_weight", 0.0)
        )
        intervention_energy = (
            positive_result["intervention_energy"]
            + negative_result["intervention_energy"]
            + cross_energy
        )
        intervention_budget_excess = (
            self._intervention_budget_excess(
                positive_result["intervention_energy"]
            )
            + self._intervention_budget_excess(
                negative_result["intervention_energy"]
            )
            + self._intervention_budget_excess(cross_energy)
        )

        total_loss = (
            default_weight * default_loss
            + positive_weight * positive_loss
            + positive_margin_weight * positive_margin_loss
            + contrastive_weight * artist_contrastive_loss
            + symmetry_weight * symmetry_loss
            + preservation_weight * preservation_loss
            + energy_weight * intervention_budget_excess
            + regularizers["loss"]
        )
        positive_gain = default_loss.detach() - positive_loss.detach()
        suppression_effect = negative_loss.detach() - default_loss.detach()
        loss_dict: Dict[str, Tensor] = {
            "default_lm_reference_monitor": default_loss,
            "positive_lm_monitor": positive_loss,
            "negative_lm_monitor": negative_loss,
            "positive_gain_monitor": positive_gain,
            "suppression_effect_monitor": suppression_effect,
            "positive_margin": positive_margin_loss,
            "wrong_positive_lm_reference_monitor": wrong_positive_loss,
            "artist_contrastive": artist_contrastive_loss,
            "artist_contrastive_pair_rate_monitor": contrastive_pair_rate,
            "suppression_symmetry": symmetry_loss,
            "cross_preservation_raw_monitor": cross_raw,
            "cross_preservation_reference_monitor": cross_reference,
            "cross_preservation_excess": preservation_loss,
            "cross_preservation_pair_rate_monitor": pair_rate,
            "cross_intervention_energy_monitor": cross_energy,
            "intervention_energy_monitor": intervention_energy,
            "intervention_budget_excess": intervention_budget_excess,
            "concept_auxiliary": regularizers["loss"],
        }
        loss_dict.update(self._scale_monitors())
        loss_dict.update(
            self._energy_monitors(positive_result, prefix="positive")
        )
        loss_dict.update(
            self._energy_monitors(negative_result, prefix="negative")
        )
        loss_dict.update(
            {
                (name if name.startswith("concept_") else f"concept_{name}"): value
                for name, value in regularizers.items()
                if name != "loss"
            }
        )
        self._log_losses(
            total_loss,
            loss_dict,
            stage=stage,
            batch_size=batch_size,
        )
        return total_loss, loss_dict

    def _scheduled_train_objective(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Train one controlled branch per step to keep activation memory flat."""

        targets = self._artist_targets(batch)
        batch_size = targets.shape[0]
        default_weight = float(
            _cfg_get(self.objective_cfg, "default_weight", 0.0)
        )
        default_context = (
            nullcontext()
            if self.train_generator and default_weight > 0
            else torch.no_grad()
        )
        with default_context:
            default_result = self._run_generator(
                batch,
                batch_idx,
                stage="train",
                condition=None,
            )
        default_losses = default_result["sample_losses"]
        default_loss = default_losses.mean()

        regularizers = self.concept_learner.compute_losses(
            style_bank_l2_weight=float(
                _cfg_get(self.loss_cfg, "style_bank_l2_weight", 0.0)
            ),
            concept_center_weight=float(
                _cfg_get(self.loss_cfg, "concept_center_weight", 0.0)
            ),
        )
        cycle = tuple(
            str(value)
            for value in _cfg_get(
                self.objective_cfg,
                "branch_cycle",
                ("positive", "symmetry", "preservation"),
            )
        )
        if not cycle or any(
            value not in {"positive", "symmetry", "preservation"}
            for value in cycle
        ):
            raise ValueError(
                "control_training.branch_cycle may contain only positive, "
                "symmetry, and preservation"
            )
        branch = cycle[batch_idx % len(cycle)]
        total_loss = regularizers["loss"] + default_weight * default_loss
        energy_weight = float(
            _cfg_get(self.loss_cfg, "intervention_energy_weight", 0.0)
        )
        loss_dict: Dict[str, Tensor] = {
            "default_lm_reference_monitor": default_loss,
            "branch_positive": default_loss.new_tensor(branch == "positive"),
            "branch_symmetry": default_loss.new_tensor(branch == "symmetry"),
            "branch_preservation": default_loss.new_tensor(
                branch == "preservation"
            ),
            "concept_auxiliary": regularizers["loss"],
        }
        loss_dict.update(self._scale_monitors())

        if branch == "positive":
            positive_result = self._run_generator(
                batch,
                batch_idx,
                stage="train",
                condition=self.concept_learner.positive_condition(targets),
            )
            positive_losses = positive_result["sample_losses"]
            positive_loss = positive_losses.mean()
            margin = float(
                _cfg_get(self.objective_cfg, "positive_margin", 0.0)
            )
            margin_loss = self._positive_margin_loss(
                positive_losses,
                default_losses,
                margin,
            )
            contrastive_cfg = _cfg_get(
                self.objective_cfg,
                "artist_contrastive",
                {},
            )
            wrong_targets = self._wrong_artist_targets(targets, batch_idx)
            contrastive_pair_rate = default_loss.new_tensor(
                wrong_targets is not None
            )
            wrong_positive_loss = default_loss.new_zeros(())
            artist_contrastive_loss = default_loss.new_zeros(())
            if (
                bool(_cfg_get(contrastive_cfg, "enabled", True))
                and wrong_targets is not None
            ):
                with torch.no_grad():
                    wrong_result = self._run_generator(
                        batch,
                        batch_idx,
                        stage="train",
                        condition=self.concept_learner.positive_condition(
                            wrong_targets
                        ),
                    )
                wrong_losses = wrong_result["sample_losses"]
                wrong_positive_loss = wrong_losses.mean()
                artist_contrastive_loss = self._artist_contrastive_loss(
                    positive_losses,
                    wrong_losses,
                    float(_cfg_get(contrastive_cfg, "margin", 0.0)),
                )
            intervention_energy = positive_result["intervention_energy"]
            intervention_budget_excess = self._intervention_budget_excess(
                intervention_energy
            )
            total_loss = (
                total_loss
                + float(_cfg_get(self.objective_cfg, "positive_weight", 0.0))
                * positive_loss
                + float(
                    _cfg_get(
                        self.objective_cfg,
                        "positive_margin_weight",
                        0.0,
                    )
                )
                * margin_loss
                + float(_cfg_get(contrastive_cfg, "weight", 0.0))
                * artist_contrastive_loss
                + energy_weight * intervention_budget_excess
            )
            loss_dict.update(
                {
                    "positive_lm_monitor": positive_loss,
                    "positive_gain_monitor": (
                        default_loss.detach() - positive_loss.detach()
                    ),
                    "positive_margin": margin_loss,
                    "wrong_positive_lm_reference_monitor": wrong_positive_loss,
                    "artist_contrastive": artist_contrastive_loss,
                    "artist_contrastive_pair_rate_monitor": contrastive_pair_rate,
                    "intervention_energy_monitor": intervention_energy,
                    "intervention_budget_excess": intervention_budget_excess,
                }
            )
            loss_dict.update(
                self._energy_monitors(positive_result, prefix="positive")
            )
        elif branch == "symmetry":
            with torch.no_grad():
                positive_result = self._run_generator(
                    batch,
                    batch_idx,
                    stage="train",
                    condition=self.concept_learner.positive_condition(targets),
                )
            negative_result = self._run_generator(
                batch,
                batch_idx,
                stage="train",
                condition=self.concept_learner.suppression_condition(targets),
            )
            positive_loss = positive_result["sample_losses"].mean()
            negative_losses = negative_result["sample_losses"]
            negative_loss = negative_losses.mean()
            symmetry_loss = self._suppression_symmetry_loss(
                default_losses,
                positive_result["sample_losses"],
                negative_losses,
            )
            intervention_energy = negative_result["intervention_energy"]
            intervention_budget_excess = self._intervention_budget_excess(
                intervention_energy
            )
            total_loss = (
                total_loss
                + float(
                    _cfg_get(
                        self.objective_cfg,
                        "suppression_symmetry_weight",
                        1.0,
                    )
                )
                * symmetry_loss
                + energy_weight * intervention_budget_excess
            )
            loss_dict.update(
                {
                    "positive_lm_reference_monitor": positive_loss,
                    "negative_lm_monitor": negative_loss,
                    "suppression_effect_monitor": (
                        negative_loss.detach() - default_loss.detach()
                    ),
                    "suppression_symmetry": symmetry_loss,
                    "intervention_energy_monitor": intervention_energy,
                    "intervention_budget_excess": intervention_budget_excess,
                }
            )
            loss_dict.update(
                self._energy_monitors(negative_result, prefix="negative")
            )
        else:
            (
                cross_raw,
                cross_reference,
                preservation_loss,
                pair_rate,
                cross_energy,
            ) = self._cross_artist_preservation(
                batch,
                batch_idx,
                stage="train",
                targets=targets,
                default_losses=default_losses,
            )
            if pair_rate.item() == 0:
                # A single-artist batch cannot form A/B preservation pairs.
                # Fall back to a useful positive update instead of producing
                # an optimizer step that only touches small regularizers.
                positive_result = self._run_generator(
                    batch,
                    batch_idx,
                    stage="train",
                    condition=self.concept_learner.positive_condition(targets),
                )
                positive_losses = positive_result["sample_losses"]
                positive_loss = positive_losses.mean()
                margin_loss = self._positive_margin_loss(
                    positive_losses,
                    default_losses,
                    float(
                        _cfg_get(
                            self.objective_cfg,
                            "positive_margin",
                            0.0,
                        )
                    ),
                )
                intervention_energy = positive_result["intervention_energy"]
                intervention_budget_excess = self._intervention_budget_excess(
                    intervention_energy
                )
                total_loss = (
                    total_loss
                    + float(
                        _cfg_get(
                            self.objective_cfg,
                            "positive_margin_weight",
                            0.0,
                        )
                    )
                    * margin_loss
                    + energy_weight * intervention_budget_excess
                )
                loss_dict.update(
                    {
                        "preservation_fallback_positive_lm_monitor": positive_loss,
                        "preservation_fallback_positive_margin": margin_loss,
                        "cross_preservation_pair_rate_monitor": pair_rate,
                        "intervention_energy_monitor": intervention_energy,
                        "intervention_budget_excess": intervention_budget_excess,
                    }
                )
                loss_dict.update(
                    self._energy_monitors(
                        positive_result,
                        prefix="preservation_fallback_positive",
                    )
                )
            else:
                preservation_cfg = _cfg_get(
                    self.objective_cfg,
                    "preservation",
                    {},
                )
                total_loss = (
                    total_loss
                    + float(_cfg_get(preservation_cfg, "weight", 1.0))
                    * preservation_loss
                    + energy_weight
                    * self._intervention_budget_excess(cross_energy)
                )
                loss_dict.update(
                    {
                        "cross_preservation_raw_monitor": cross_raw,
                        "cross_preservation_reference_monitor": cross_reference,
                        "cross_preservation_excess": preservation_loss,
                        "cross_preservation_pair_rate_monitor": pair_rate,
                        "cross_intervention_energy_monitor": cross_energy,
                        "intervention_energy_monitor": cross_energy,
                        "intervention_budget_excess": (
                            self._intervention_budget_excess(cross_energy)
                        ),
                    }
                )

        loss_dict.update(
            {
                (name if name.startswith("concept_") else f"concept_{name}"): value
                for name, value in regularizers.items()
                if name != "loss"
            }
        )
        self._log_losses(
            total_loss,
            loss_dict,
            stage="train",
            batch_size=batch_size,
        )
        return total_loss, loss_dict

    def _controlled_shared_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        *,
        stage: str,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if stage == "train":
            return self._scheduled_train_objective(batch, batch_idx)
        return self._full_control_objective(batch, batch_idx, stage=stage)

    def _explicit_condition(
        self,
        batch: Mapping[str, Any],
        *,
        default_direction: float = 0.0,
    ) -> Optional[ConceptCondition]:
        suppressed = batch.get("suppressed_artist_ids")
        if suppressed is not None:
            return self.concept_learner.suppression_condition(suppressed)
        direction = batch.get("control_direction", default_direction)
        direction_tensor = torch.as_tensor(direction)
        if direction_tensor.numel() == 1 and float(direction_tensor) == 0.0:
            return None
        targets = self._artist_targets(batch)
        return self.concept_learner.prepare_condition(
            concept_ids=targets,
            direction=direction,
        )

    def forward(self, batch: Mapping[str, Any], **kwargs):
        condition = self._explicit_condition(batch)
        if condition is None:
            return self.model(self._generator_batch(batch), **kwargs)
        handles = self.model.register_concept_learner_hooks(
            self.concept_learner,
            condition=condition,
            module_names=self.hook_blocks,
            intervention_scale=self.intervention_scale,
            batch_repeat_interleave=int(getattr(self.model, "num_codebooks", 1) or 1),
        )
        try:
            return self.model(self._generator_batch(batch), **kwargs)
        finally:
            self.model.remove_hidden_state_hooks(handles)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        total_loss, _ = self._controlled_shared_step(
            batch,
            batch_idx,
            stage="train",
        )
        return total_loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Tensor]:
        del dataloader_idx
        device = next(
            (
                value.device
                for value in batch.values()
                if isinstance(value, Tensor)
            ),
            self.device,
        )
        precision_context = nullcontext()
        if self.validation_full_precision and device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", enabled=False)
        with precision_context:
            total_loss, loss_dict = self._controlled_shared_step(
                batch,
                batch_idx,
                stage="val",
            )
        return {
            "loss": total_loss.detach(),
            **{name: value.detach() for name, value in loss_dict.items()},
        }

    @torch.no_grad()
    def _generate_with_condition(
        self,
        batch: Mapping[str, Any],
        generation_kwargs: Mapping[str, Any],
        condition: Optional[ConceptCondition] = None,
    ) -> Tensor:
        controlled = dict(batch)
        controlled["generation_kwargs"] = dict(generation_kwargs)
        handles = []
        if condition is not None:
            handles = self.model.register_concept_learner_hooks(
                self.concept_learner,
                condition=condition,
                module_names=self.hook_blocks,
                intervention_scale=self.intervention_scale,
                batch_repeat_interleave=int(
                    getattr(self.model, "num_codebooks", 1) or 1
                ),
                cfg_conditional_only=self._generation_uses_cfg(controlled),
            )
        try:
            audio, _ = self._generate_test_batch(self._generator_batch(controlled))
        finally:
            if handles:
                self.model.remove_hidden_state_hooks(handles)
        return audio

    @torch.no_grad()
    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        del dataloader_idx
        condition = self._explicit_condition(batch)
        generation_kwargs = batch.get("generation_kwargs") or {}
        audio = self._generate_with_condition(batch, generation_kwargs, condition)
        output = self._test_output(audio, self._test_prompts(batch), batch_idx)
        if condition is not None:
            output["concept_weights"] = condition.weights.detach()
            output["control_direction"] = condition.direction.detach()
        return output

    @torch.no_grad()
    def generate_validation_comparisons(
        self,
        batch: Mapping[str, Any],
        *,
        counterfactual_batch: Optional[Mapping[str, Any]] = None,
        generation_kwargs: Optional[Mapping[str, Any]] = None,
        generation_seed: int = 42,
    ) -> Dict[str, Any]:
        """Compare default, positive, and explicit negative artist controls."""

        audio_tokens = batch.get("audio_tokens")
        if not isinstance(audio_tokens, Tensor):
            raise ValueError("validation comparison requires audio_tokens")
        targets = self._artist_targets(batch)
        ground_truth, ground_truth_lengths = self.model.decode_audio_tokens(
            audio_tokens,
            batch.get("decoder_attention_mask"),
        )
        other_ground_truth = None
        other_ground_truth_lengths = None
        if counterfactual_batch is not None:
            other_tokens = counterfactual_batch.get("audio_tokens")
            if not isinstance(other_tokens, Tensor):
                raise ValueError("counterfactual comparison requires audio_tokens")
            other_ground_truth, other_ground_truth_lengths = self.model.decode_audio_tokens(
                other_tokens,
                counterfactual_batch.get("decoder_attention_mask"),
            )

        kwargs = dict(generation_kwargs or {})
        device = audio_tokens.device
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index if device.index is not None else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(generation_seed)
            generated = self._generate_with_condition(batch, kwargs, None)
            torch.manual_seed(generation_seed)
            positive = self._generate_with_condition(
                batch,
                kwargs,
                self.concept_learner.positive_condition(targets),
            )
            torch.manual_seed(generation_seed)
            suppressed = self._generate_with_condition(
                batch,
                kwargs,
                self.concept_learner.suppression_condition(targets),
            )
            other_generated = None
            suppressed_with_other_caption = None
            if counterfactual_batch is not None:
                torch.manual_seed(generation_seed)
                other_generated = self._generate_with_condition(
                    counterfactual_batch,
                    kwargs,
                    None,
                )
                torch.manual_seed(generation_seed)
                suppressed_with_other_caption = self._generate_with_condition(
                    counterfactual_batch,
                    kwargs,
                    self.concept_learner.suppression_condition(targets),
                )

        output: Dict[str, Any] = {
            "ground_truth": ground_truth,
            "ground_truth_lengths": ground_truth_lengths,
            "generated": generated,
            "positive": positive,
            "suppressed": suppressed,
            "prompts": self._test_prompts(batch),
            "metadata": batch.get("metadata"),
            "sample_rate": int(getattr(self.model, "audio_sample_rate", 32000)),
        }
        if counterfactual_batch is not None:
            output.update(
                {
                    "suppressed_with_other_caption": suppressed_with_other_caption,
                    "other_ground_truth": other_ground_truth,
                    "other_ground_truth_lengths": other_ground_truth_lengths,
                    "other_generated": other_generated,
                    "other_prompts": self._test_prompts(counterfactual_batch),
                    "other_metadata": counterfactual_batch.get("metadata"),
                }
            )
        return output


UnlearnableGenModel = UnlearnableGenerationModel
UnlearnableMusicGenLightningModule = UnlearnableGenerationModel

__all__ = [
    "UnlearnableGenModel",
    "UnlearnableGenerationModel",
    "UnlearnableMusicGenLightningModule",
]
