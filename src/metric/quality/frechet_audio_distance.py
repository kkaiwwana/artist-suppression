"""Frechet Audio Distance (FAD)."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn
from torchmetrics import Metric

from ._audio import extract_tensor_output, to_mono_batch


class MelStatsAudioEmbedding(nn.Module):
    """Small offline audio embedding used as a dependency-light fallback.

    This is not the canonical FAD embedding. For publication-quality FAD, pass a
    VGGish/PANNs/MERT/music-audio embedding model through ``embedding_model``.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
    ) -> None:
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError(
                "torchaudio is required for MelStatsAudioEmbedding fallback."
            ) from exc

        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, audio: Tensor, sample_rate: int | None = None) -> Tensor:
        waveform = to_mono_batch(audio)
        if sample_rate is not None and sample_rate != self.sample_rate:
            try:
                import torchaudio

                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sample_rate, new_freq=self.sample_rate
                )
            except ImportError as exc:
                raise ImportError("torchaudio is required for resampling.") from exc
        mel = self.melspec(waveform)
        log_mel = torch.log(mel + 1e-6)
        return torch.cat([log_mel.mean(dim=-1), log_mel.std(dim=-1)], dim=-1)


class FrechetAudioDistance(Metric):
    """Frechet distance between generated and reference audio embeddings."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        embedding_model: Any | None = None,
        *,
        embedding_encoder: Callable[..., Tensor] | None = None,
        sample_rate: int = 16_000,
        use_mel_fallback: bool = True,
        embedding_key: str = "embeddings",
        model_kwargs: dict[str, Any] | None = None,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if embedding_model is None and embedding_encoder is None and use_mel_fallback:
            embedding_model = MelStatsAudioEmbedding(sample_rate=sample_rate)
        self.embedding_model = embedding_model
        self.embedding_encoder = embedding_encoder
        self.sample_rate = sample_rate
        self.embedding_key = embedding_key
        self.model_kwargs = dict(model_kwargs or {})
        self.eps = eps

        self.add_state("generated_embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("reference_embeddings", default=[], dist_reduce_fx="cat")

    def _embed(self, audio: Tensor, sample_rate: int | None) -> Tensor:
        if self.embedding_encoder is None and self.embedding_model is None:
            raise RuntimeError(
                "FrechetAudioDistance needs an embedding_model/embedding_encoder, "
                "or use_mel_fallback=True."
            )

        with torch.no_grad():
            if self.embedding_encoder is not None:
                try:
                    output = self.embedding_encoder(audio, sample_rate=sample_rate, **self.model_kwargs)
                except TypeError:
                    output = self.embedding_encoder(audio, **self.model_kwargs)
            else:
                model = self.embedding_model
                if hasattr(model, "eval"):
                    model.eval()
                if hasattr(model, "encode_audio"):
                    try:
                        output = model.encode_audio(audio, sample_rate=sample_rate, **self.model_kwargs)
                    except TypeError:
                        output = model.encode_audio(audio, **self.model_kwargs)
                elif hasattr(model, "get_audio_features"):
                    output = model.get_audio_features(audio, **self.model_kwargs)
                else:
                    try:
                        output = model(audio, sample_rate=sample_rate, **self.model_kwargs)
                    except TypeError:
                        output = model(audio, **self.model_kwargs)

        embeddings = extract_tensor_output(output, preferred_key=self.embedding_key)
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim > 2:
            embeddings = embeddings.flatten(start_dim=1)
        return embeddings.float()

    def update(
        self,
        generated_audio: Tensor | None = None,
        reference_audio: Tensor | None = None,
        *,
        sample_rate: int | None = None,
    ) -> None:
        sr = sample_rate if sample_rate is not None else self.sample_rate
        if generated_audio is not None:
            self.update_generated(generated_audio, sample_rate=sr)
        if reference_audio is not None:
            self.update_reference(reference_audio, sample_rate=sr)

    def update_generated(self, audio: Tensor, *, sample_rate: int | None = None) -> None:
        embeddings = self._embed(audio, sample_rate if sample_rate is not None else self.sample_rate)
        self.generated_embeddings.append(embeddings.detach())

    def update_reference(self, audio: Tensor, *, sample_rate: int | None = None) -> None:
        embeddings = self._embed(audio, sample_rate if sample_rate is not None else self.sample_rate)
        self.reference_embeddings.append(embeddings.detach())

    @staticmethod
    def _cat_state(state: list[Tensor] | Tensor) -> Tensor | None:
        if isinstance(state, Tensor):
            return state
        if len(state) == 0:
            return None
        return torch.cat(state, dim=0)

    @staticmethod
    def _covariance(features: Tensor) -> Tensor:
        if features.shape[0] <= 1:
            return torch.zeros(
                features.shape[1],
                features.shape[1],
                device=features.device,
                dtype=features.dtype,
            )
        centered = features - features.mean(dim=0, keepdim=True)
        return centered.T @ centered / (features.shape[0] - 1)

    @staticmethod
    def _matrix_sqrt_psd(matrix: Tensor, eps: float) -> Tensor:
        matrix = (matrix + matrix.T) / 2.0
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = eigvals.clamp_min(eps).sqrt()
        return (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T

    def compute(self) -> Tensor:
        generated = self._cat_state(self.generated_embeddings)
        reference = self._cat_state(self.reference_embeddings)
        if generated is None or reference is None:
            return torch.tensor(float("nan"))
        if generated.shape[1] != reference.shape[1]:
            raise ValueError(
                "Generated and reference embeddings must have the same dimension; "
                f"got {generated.shape[1]} and {reference.shape[1]}."
            )

        generated = generated.float()
        reference = reference.to(generated.device).float()
        mu_generated = generated.mean(dim=0)
        mu_reference = reference.mean(dim=0)
        cov_generated = self._covariance(generated)
        cov_reference = self._covariance(reference)

        mean_term = (mu_generated - mu_reference).pow(2).sum()
        sqrt_cov_generated = self._matrix_sqrt_psd(cov_generated, self.eps)
        cov_mean = self._matrix_sqrt_psd(
            sqrt_cov_generated @ cov_reference @ sqrt_cov_generated,
            self.eps,
        )
        trace_term = torch.trace(cov_generated + cov_reference - 2.0 * cov_mean)
        return (mean_term + trace_term).clamp_min(0.0)
