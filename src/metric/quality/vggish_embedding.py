"""Canonical pre-activation VGGish embeddings for Frechet Audio Distance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from ._audio import extract_tensor_output, to_mono_batch


def _load_default_vggish() -> nn.Module:
    """Load torchvggish and its public checkpoint through Torch Hub."""

    try:
        return torch.hub.load(
            "harritaylor/torchvggish",
            "vggish",
            trust_repo=True,
            preprocess=False,
            postprocess=False,
            device="cpu",
        )
    except RuntimeError as exc:
        # torch.hub reports a missing package from hubconf.py as RuntimeError,
        # while genuine download/checkpoint errors should retain their details.
        if "Missing dependencies" not in str(exc):
            raise
        raise ImportError(
            "VGGishAudioEmbedding needs the optional torchvggish dependencies. "
            "Install `resampy` and `soundfile`, or inject model/model_loader and "
            "a preprocessor."
        ) from exc
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "VGGishAudioEmbedding needs the optional torchvggish dependencies. "
            "Install `resampy` and `soundfile`, or inject model/model_loader and "
            "a preprocessor."
        ) from exc


class VGGishAudioEmbedding(nn.Module):
    """Extract canonical 128-D VGGish windows while preserving clip groups.

    The default backend is the TensorFlow-compatible ``torchvggish`` port.  It
    is loaded lazily from Torch Hub, its PCA/quantization postprocessing is
    disabled, and its final ReLU is removed so that outputs are the canonical
    pre-activation embeddings used by FAD.

    ``forward`` returns all windows concatenated as ``[num_windows, 128]``.
    ``encode_with_clip_ids`` additionally identifies the source clip of every
    row, while ``encode_audio``/``encode_grouped`` return one window matrix per
    clip.  This lets callers pool all windows belonging to an artist without
    first averaging away the within-clip distribution.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_loader: Callable[[], Any] | None = None,
        preprocessor: Callable[..., Any] | None = None,
        sample_rate: int = 16_000,
        peak_normalize: bool = True,
        remove_final_relu: bool = True,
        window_batch_size: int = 64,
        lazy_load: bool = True,
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive.")

        self.model = model
        self.model_loader = model_loader
        self.preprocessor = preprocessor
        self.sample_rate = int(sample_rate)
        self.peak_normalize = bool(peak_normalize)
        self.remove_final_relu = bool(remove_final_relu)
        self.window_batch_size = int(window_batch_size)
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

        if self.model is not None:
            self._configure_model(self.model)
        elif not lazy_load:
            self._ensure_model()

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _configure_model(self, model: Any) -> Any:
        # torchvggish performs NumPy preprocessing itself by default and applies
        # PCA/8-bit quantization by default.  This wrapper owns preprocessing and
        # needs the raw network output instead.
        if hasattr(model, "preprocess"):
            model.preprocess = False
        if hasattr(model, "postprocess"):
            model.postprocess = False

        embeddings = getattr(model, "embeddings", None)
        if (
            self.remove_final_relu
            and isinstance(embeddings, nn.Sequential)
            and len(embeddings) > 0
            and isinstance(embeddings[-1], nn.ReLU)
        ):
            embeddings[-1] = nn.Identity()

        if hasattr(model, "eval"):
            model.eval()
        if isinstance(model, nn.Module):
            model.requires_grad_(False)
            model.to(self.device)
        # torchvggish keeps a separate device attribute used inside forward.
        if hasattr(model, "device"):
            model.device = self.device
        return model

    def _ensure_model(self) -> Any:
        if self.model is None:
            loader = self.model_loader or _load_default_vggish
            self.model = loader()
            if self.model is None:
                raise RuntimeError("model_loader returned None.")
        return self._configure_model(self.model)

    def _normalize_waveforms(self, waveform: Tensor) -> Tensor:
        if not self.peak_normalize:
            return waveform
        # Match Google's FAD AudioSet wrapper: x / max(0.1, max(x)) for each
        # clip.  This is deliberately the positive peak, rather than abs().max().
        scale = waveform.amax(dim=-1, keepdim=True).clamp_min(0.1)
        return waveform / scale

    @staticmethod
    def _call_preprocessor(
        preprocessor: Callable[..., Any], waveform: Any, sample_rate: int
    ) -> Any:
        try:
            return preprocessor(waveform, sample_rate)
        except TypeError as positional_error:
            try:
                return preprocessor(waveform, sample_rate=sample_rate)
            except TypeError:
                raise positional_error

    def _preprocess_clip(
        self, waveform: Tensor, sample_rate: int, model: Any
    ) -> Tensor:
        preprocessor = self.preprocessor
        if preprocessor is None:
            preprocessor = getattr(model, "_preprocess", None)
        if preprocessor is None:
            raise RuntimeError(
                "The injected VGGish model has no `_preprocess` method. Pass a "
                "preprocessor that converts one waveform into VGGish windows."
            )

        examples = self._call_preprocessor(
            preprocessor, waveform.detach().float().cpu().numpy(), sample_rate
        )
        examples = (
            examples if isinstance(examples, Tensor) else torch.as_tensor(examples)
        )
        examples = examples.detach().float()
        # Canonical torchvggish preprocessing returns [windows, 1, 96, 64].
        # Accept [windows, 96, 64] from equivalent preprocessors as well.
        if examples.ndim == 3:
            examples = examples.unsqueeze(1)
        if examples.ndim < 2:
            raise ValueError(
                "VGGish preprocessor must return [windows, ...]; "
                f"got {tuple(examples.shape)}."
            )
        return examples

    def _run_model(self, examples: Tensor, batch_size: int) -> Tensor:
        model = self._ensure_model()
        outputs: list[Tensor] = []
        for start in range(0, examples.shape[0], batch_size):
            window_batch = examples[start : start + batch_size].to(
                device=self.device, dtype=torch.float32
            )
            output = model(window_batch)
            embeddings = extract_tensor_output(output, preferred_key="embeddings")
            embeddings = embeddings.to(device=self.device)
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            if embeddings.ndim > 2:
                embeddings = embeddings.flatten(start_dim=1)
            if embeddings.ndim != 2 or embeddings.shape[0] != window_batch.shape[0]:
                raise ValueError(
                    "VGGish model must return [windows, embedding_dim]; "
                    f"got {tuple(embeddings.shape)} for "
                    f"{window_batch.shape[0]} windows."
                )
            outputs.append(embeddings.float())
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def encode_with_clip_ids(
        self,
        audio: Tensor,
        sample_rate: int | None = None,
        *,
        batch_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return concatenated window embeddings and their zero-based clip ids."""

        source_rate = self.sample_rate if sample_rate is None else int(sample_rate)
        if source_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        inference_batch_size = (
            self.window_batch_size if batch_size is None else int(batch_size)
        )
        if inference_batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        waveform = to_mono_batch(audio)
        if waveform.shape[-1] == 0:
            raise ValueError("Audio clips must contain at least one sample.")
        waveform = self._normalize_waveforms(waveform)
        model = self._ensure_model()

        windows: list[Tensor] = []
        clip_ids: list[Tensor] = []
        for clip_id, clip in enumerate(waveform):
            examples = self._preprocess_clip(clip, source_rate, model)
            if examples.shape[0] == 0:
                raise ValueError(
                    "VGGish preprocessing produced no ~0.96-second window for "
                    f"clip {clip_id}."
                )
            windows.append(examples)
            clip_ids.append(torch.full((examples.shape[0],), clip_id, dtype=torch.long))

        all_windows = torch.cat(windows, dim=0)
        embeddings = self._run_model(all_windows, inference_batch_size)
        ids = torch.cat(clip_ids, dim=0).to(device=embeddings.device)
        return embeddings, ids

    @torch.inference_mode()
    def encode_audio(
        self,
        audio: Tensor,
        sample_rate: int | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[Tensor]:
        """Return one ``[windows, embedding_dim]`` tensor per input clip."""

        waveform = to_mono_batch(audio)
        embeddings, clip_ids = self.encode_with_clip_ids(
            waveform, sample_rate=sample_rate, batch_size=batch_size
        )
        return [embeddings[clip_ids == index] for index in range(waveform.shape[0])]

    def encode_grouped(
        self,
        audio: Tensor,
        sample_rate: int | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[Tensor]:
        """Alias for :meth:`encode_audio`, emphasizing clip grouping."""

        return self.encode_audio(audio, sample_rate=sample_rate, batch_size=batch_size)

    def forward(self, audio: Tensor, sample_rate: int | None = None) -> Tensor:
        embeddings, _ = self.encode_with_clip_ids(audio, sample_rate=sample_rate)
        return embeddings


__all__ = ["VGGishAudioEmbedding"]
