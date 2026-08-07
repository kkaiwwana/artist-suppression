"""Runtime adapters and per-clip statistics for epoch control evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.evaluation.checkpoint_runtime import (
    locate_checkpoint_config,
    trusted_torch_load,
)


@dataclass(frozen=True)
class ClassifierTargetStatistics:
    """Per-clip statistics for the cohort's intended artist class."""

    attribution: Tensor
    confidence: Tensor
    rank: Tensor


def classifier_target_statistics(
    logits: Tensor,
    target_indices: Tensor | Sequence[int],
) -> ClassifierTargetStatistics:
    """Compute top-1 attribution, target confidence, and one-based rank."""

    values = torch.as_tensor(logits).float()
    targets = torch.as_tensor(target_indices, device=values.device).long()
    if values.ndim != 2 or targets.ndim != 1 or values.shape[0] != targets.numel():
        raise ValueError("logits [B,C] and target_indices [B] must align")
    if values.shape[1] <= 1:
        raise ValueError("classifier logits need at least two classes")
    if bool(((targets < 0) | (targets >= values.shape[1])).any()):
        raise ValueError("target_indices contain an out-of-range class")
    target_logits = values.gather(1, targets.unsqueeze(1)).squeeze(1)
    probabilities = values.softmax(dim=-1)
    confidence = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    rank = 1 + (values > target_logits.unsqueeze(1)).sum(dim=1)
    attribution = values.argmax(dim=-1).eq(targets).float()
    return ClassifierTargetStatistics(
        attribution=attribution.cpu(),
        confidence=confidence.cpu(),
        rank=rank.float().cpu(),
    )


class ArtistClassifierRuntime:
    """Load a trained MERT artist classifier and classify generated audio."""

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        artist_vocabulary: Sequence[str],
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        vocabulary = tuple(str(value) for value in artist_vocabulary)
        if not vocabulary or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("artist_vocabulary must be non-empty and unique")
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.processor = processor
        self.artist_vocabulary = vocabulary
        self.artist_to_index = {
            value: index for index, value in enumerate(self.artist_vocabulary)
        }
        self.sample_rate = int(getattr(processor, "sampling_rate", 24_000))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config_path: str | Path | None = None,
        model_name_or_path: str | Path | None = None,
        local_files_only: bool = False,
        device: torch.device | str = "cpu",
    ) -> "ArtistClassifierRuntime":
        """Reconstruct the classifier using the checkpoint's Hydra config."""

        from omegaconf import OmegaConf, open_dict
        from transformers import Wav2Vec2FeatureExtractor

        from src.runner import setup_model

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        resolved_config = locate_checkpoint_config(checkpoint, config_path)
        config = OmegaConf.load(resolved_config)
        payload = trusted_torch_load(checkpoint)
        vocabulary = tuple(str(value) for value in payload.get("artist_vocabulary", ()))
        if not vocabulary:
            raise ValueError(
                "artist classifier checkpoint has no artist_vocabulary metadata"
            )

        classifier_cfg = config.runner.model.model_cfg.classifier
        resolved_model_path = str(
            model_name_or_path or classifier_cfg.model_name_or_path
        )
        with open_dict(classifier_cfg):
            classifier_cfg.num_classes = len(vocabulary)
            classifier_cfg.model_name_or_path = resolved_model_path
            classifier_cfg.local_files_only = bool(local_files_only)
        with open_dict(config.runner.model.model_cfg):
            config.runner.model.model_cfg.validation_names = ["gt", "default"]

        lightning_module = setup_model(config)
        state = payload.get("state_dict", payload)
        incompatible = lightning_module.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "classifier checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            local_files_only=bool(local_files_only),
        )
        return cls(
            lightning_module.model,
            processor,
            vocabulary,
            device=device,
        )

    @staticmethod
    def _mono(audio: Tensor) -> Tensor:
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        if audio.ndim != 2:
            raise ValueError("audio must have shape [B,T] or [B,C,T]")
        return audio.detach().float().cpu()

    @torch.inference_mode()
    def logits(
        self,
        audio: Tensor,
        *,
        sample_rate: int,
        batch_size: int = 8,
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        waveforms = self._mono(audio)
        if int(sample_rate) != self.sample_rate:
            import torchaudio

            waveforms = torchaudio.functional.resample(
                waveforms, int(sample_rate), self.sample_rate
            )
        outputs: list[Tensor] = []
        for start in range(0, waveforms.shape[0], batch_size):
            values = [row.numpy() for row in waveforms[start : start + batch_size]]
            encoded = self.processor(
                values,
                sampling_rate=self.sample_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_values = (
                torch.as_tensor(encoded["input_values"]).float().to(self.device)
            )
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = torch.as_tensor(attention_mask).long().to(self.device)
            outputs.append(
                self.model(input_values, attention_mask).detach().float().cpu()
            )
        return torch.cat(outputs, dim=0)

    def classifier_indices(self, artist_keys: Sequence[str]) -> Tensor:
        missing = sorted(set(artist_keys).difference(self.artist_to_index))
        if missing:
            raise KeyError(
                "artists are absent from classifier vocabulary: " + ", ".join(missing)
            )
        return torch.tensor(
            [self.artist_to_index[value] for value in artist_keys], dtype=torch.long
        )

    def to(self, device: torch.device | str) -> "ArtistClassifierRuntime":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self


def cosine_similarity_rows(first: Tensor, second: Tensor) -> Tensor:
    """Cosine similarity for aligned embedding rows."""

    first = torch.as_tensor(first).float()
    second = torch.as_tensor(second).float()
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("embedding matrices must have matching [B,D] shapes")
    return (F.normalize(first, dim=-1) * F.normalize(second, dim=-1)).sum(dim=-1)


def probability_kl_divergence(
    reference_probabilities: Tensor,
    generated_probabilities: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Per-clip ``KL(reference || generated)`` over AudioSet labels."""

    reference = torch.as_tensor(reference_probabilities).float()
    generated = torch.as_tensor(generated_probabilities).float()
    if reference.ndim != 2 or generated.shape != reference.shape:
        raise ValueError("probability matrices must have matching [B,C] shapes")
    if eps <= 0:
        raise ValueError("eps must be positive")
    reference = reference.clamp_min(eps)
    generated = generated.clamp_min(eps)
    reference = reference / reference.sum(dim=-1, keepdim=True).clamp_min(eps)
    generated = generated / generated.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (reference * (reference.log() - generated.log())).sum(dim=-1)


def frechet_distance_from_embeddings(
    generated: Tensor,
    reference: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Fréchet distance between two embedding distributions."""

    generated = torch.as_tensor(generated).double()
    reference = torch.as_tensor(reference).double()
    if generated.ndim != 2 or reference.ndim != 2:
        raise ValueError("embeddings must have shape [N,D]")
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("generated/reference embedding dimensions must match")
    if generated.shape[0] < 2 or reference.shape[0] < 2:
        raise ValueError("FAD needs at least two samples in each distribution")

    def covariance(values: Tensor) -> Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        return centered.T @ centered / (values.shape[0] - 1)

    def matrix_sqrt_psd(matrix: Tensor) -> Tensor:
        matrix = (matrix + matrix.T) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        return (
            eigenvectors * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(0)
        ) @ eigenvectors.T

    generated_mean = generated.mean(dim=0)
    reference_mean = reference.mean(dim=0)
    generated_covariance = covariance(generated)
    reference_covariance = covariance(reference)
    generated_sqrt = matrix_sqrt_psd(generated_covariance)
    covariance_mean = matrix_sqrt_psd(
        generated_sqrt @ reference_covariance @ generated_sqrt
    )
    result = (generated_mean - reference_mean).square().sum() + torch.trace(
        generated_covariance + reference_covariance - 2.0 * covariance_mean
    )
    return result.clamp_min(0.0).float()


__all__ = [
    "ArtistClassifierRuntime",
    "ClassifierTargetStatistics",
    "classifier_target_statistics",
    "cosine_similarity_rows",
    "frechet_distance_from_embeddings",
    "probability_kl_divergence",
]
