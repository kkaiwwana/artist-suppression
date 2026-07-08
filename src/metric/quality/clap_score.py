"""CLAP-style audio-text alignment score."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

from ._audio import extract_tensor_output, match_batch


class CLAPScore(Metric):
    """Mean cosine similarity between audio and text embeddings.

    A real evaluation should pass a pretrained CLAP model or explicit
    ``audio_encoder``/``text_encoder`` callables. If ``load_default_model=True``,
    the metric tries to lazy-load ``laion_clap.CLAP_Module`` and download its
    checkpoint when first used.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        model: Any | None = None,
        *,
        audio_encoder: Callable[..., Tensor] | None = None,
        text_encoder: Callable[..., Tensor] | None = None,
        sample_rate: int | None = None,
        load_default_model: bool = False,
        scale: float = 1.0,
        clamp_min: float | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.sample_rate = sample_rate
        self.load_default_model = load_default_model
        self.scale = float(scale)
        self.clamp_min = clamp_min
        self.model_kwargs = dict(model_kwargs or {})

        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        if not self.load_default_model:
            raise RuntimeError(
                "CLAPScore needs a model/audio_encoder+text_encoder, or set "
                "load_default_model=True to try laion_clap lazy loading."
            )
        try:
            import laion_clap  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "laion_clap is not installed. Install it or pass custom encoders to CLAPScore."
            ) from exc
        device = self.score_sum.device
        model = laion_clap.CLAP_Module(enable_fusion=False, device=str(device))
        model.load_ckpt()
        self.model = model
        return model

    @staticmethod
    def _as_text_list(text: str | Sequence[str]) -> list[str]:
        if isinstance(text, str):
            return [text]
        return list(text)

    def _encode_audio(self, audio: Tensor, sample_rate: int | None) -> Tensor:
        with torch.no_grad():
            if self.audio_encoder is not None:
                try:
                    output = self.audio_encoder(audio, sample_rate=sample_rate, **self.model_kwargs)
                except TypeError:
                    output = self.audio_encoder(audio, **self.model_kwargs)
            else:
                model = self._ensure_model()
                if hasattr(model, "encode_audio"):
                    try:
                        output = model.encode_audio(audio, sample_rate=sample_rate, **self.model_kwargs)
                    except TypeError:
                        output = model.encode_audio(audio, **self.model_kwargs)
                elif hasattr(model, "get_audio_features"):
                    output = model.get_audio_features(audio, **self.model_kwargs)
                elif hasattr(model, "get_audio_embedding_from_data"):
                    try:
                        output = model.get_audio_embedding_from_data(
                            x=audio.detach().cpu().numpy(), use_tensor=True
                        )
                    except TypeError:
                        output = model.get_audio_embedding_from_data(audio, use_tensor=True)
                else:
                    try:
                        output = model(audio, sample_rate=sample_rate, **self.model_kwargs)
                    except TypeError:
                        output = model(audio, **self.model_kwargs)
        embeddings = extract_tensor_output(output, preferred_key="audio_embeds").to(audio.device)
        if embeddings.ndim > 2:
            embeddings = embeddings.mean(dim=tuple(range(1, embeddings.ndim - 1)))
        return embeddings.float()

    def _encode_text(self, text: str | Sequence[str], device: torch.device) -> Tensor:
        text_list = self._as_text_list(text)
        with torch.no_grad():
            if self.text_encoder is not None:
                output = self.text_encoder(text_list, **self.model_kwargs)
            else:
                model = self._ensure_model()
                if hasattr(model, "encode_text"):
                    output = model.encode_text(text_list, **self.model_kwargs)
                elif hasattr(model, "get_text_features"):
                    output = model.get_text_features(text_list, **self.model_kwargs)
                elif hasattr(model, "get_text_embedding"):
                    output = model.get_text_embedding(text_list, use_tensor=True)
                else:
                    output = model(text_list, **self.model_kwargs)
        embeddings = extract_tensor_output(output, preferred_key="text_embeds").to(device)
        if embeddings.ndim > 2:
            embeddings = embeddings.mean(dim=tuple(range(1, embeddings.ndim - 1)))
        return embeddings.float()

    def update(
        self,
        audio: Tensor,
        text: str | Sequence[str],
        *,
        sample_rate: int | None = None,
    ) -> None:
        audio_embeddings = self._encode_audio(
            audio, sample_rate if sample_rate is not None else self.sample_rate
        )
        text_embeddings = self._encode_text(text, audio_embeddings.device)
        audio_embeddings, text_embeddings = match_batch(audio_embeddings, text_embeddings)

        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        scores = (audio_embeddings * text_embeddings).sum(dim=-1) * self.scale
        if self.clamp_min is not None:
            scores = scores.clamp_min(self.clamp_min)

        self.score_sum += scores.sum()
        self.total += torch.tensor(float(scores.numel()), device=self.total.device)

    def compute(self) -> Tensor:
        return self.score_sum / self.total.clamp_min(1.0)
