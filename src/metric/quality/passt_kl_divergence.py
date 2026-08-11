"""PaSST AudioSet KL divergence used by MusicGen/AudioCraft evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics import Metric

from src.evaluation._external_output import quiet_external_output

from ._audio import match_batch, to_mono_batch


_STFT_PATCH_LOCK = RLock()


@contextmanager
def _legacy_stft_compatibility(enabled: bool) -> Iterator[None]:
    """Temporarily provide the real-valued STFT output expected by PaSST.

    ``hear21passt`` still calls the pre-complex PyTorch STFT API.  AudioCraft
    applies the same compatibility patch around PaSST inference.  The lock and
    ``finally`` block keep the process-wide monkey patch scoped and make sure it
    is restored even when model inference raises.
    """

    if not enabled:
        yield
        return

    with _STFT_PATCH_LOCK:
        original_stft = torch.stft

        def legacy_stft(*args: Any, **kwargs: Any) -> Tensor:
            # ``return_complex`` is the tenth positional parameter. Respect an
            # explicit value supplied by a newer classifier and only provide
            # the legacy default when it is genuinely omitted.
            if len(args) < 10:
                kwargs.setdefault("return_complex", False)
            return original_stft(*args, **kwargs)

        torch.stft = legacy_stft  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.stft = original_stft  # type: ignore[assignment]


def _load_default_passt_classifier() -> nn.Module:
    """Load the public 527-class PaSST AudioSet logits model on demand."""

    try:
        from hear21passt.base import get_basic_model  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PaSSTKLDivergence needs the optional PaSST dependency. Install "
            "`hear21passt` and `timm`, or inject classifier/classifier_loader."
        ) from exc
    # hear21passt prints the full network repr while constructing the public
    # model. That is useful interactively but overwhelms epoch-evaluation logs.
    with quiet_external_output():
        return get_basic_model(mode="logits")


def _extract_logits(output: Any) -> Tensor:
    """Extract a two-dimensional logits tensor from common model outputs."""

    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        logits = None
        for key in ("logits", "clipwise_output", "output"):
            if key in output:
                logits = output[key]
                break
        if logits is None:
            raise ValueError(
                "PaSST classifier mapping output must contain `logits`, "
                "`clipwise_output`, or `output`."
            )
        logits = logits if isinstance(logits, Tensor) else torch.as_tensor(logits)
    elif hasattr(output, "logits"):
        logits = output.logits
        logits = logits if isinstance(logits, Tensor) else torch.as_tensor(logits)
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
        logits = logits if isinstance(logits, Tensor) else torch.as_tensor(logits)
    else:
        logits = torch.as_tensor(output)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(
            "PaSST classifier must return logits with shape [batch, classes]; "
            f"got {tuple(logits.shape)}."
        )
    if logits.shape[1] < 2:
        raise ValueError("PaSST classifier must return at least two classes.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("PaSST classifier returned non-finite logits.")
    return logits


class PaSSTKLDivergence(Metric):
    """MusicGen-style KL divergence between paired audio clips.

    Each generated/reference clip is converted to mono and resampled to 32 kHz,
    then classified with the 527-class AudioSet PaSST model.  Following
    AudioCraft, classifier logits are normalized with softmax and the primary
    score is ``KL(reference || generated)``.  The reverse and the sum of both
    directions are reported as well.

    The default classifier is loaded only on the first update and may download
    its public checkpoint.  Tests and offline callers can inject a classifier or
    a zero-argument ``classifier_loader`` instead.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        classifier: Any | None = None,
        *,
        classifier_loader: Callable[[], Any] | None = None,
        sample_rate: int = 32_000,
        max_duration_seconds: float = 10.0,
        patch_torch_stft: bool = True,
        epsilon: float = 1e-6,
        lazy_load: bool = True,
        quiet_backend: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.classifier = classifier
        self.classifier_loader = classifier_loader
        self.sample_rate = int(sample_rate)
        self.max_duration_seconds = float(max_duration_seconds)
        self.patch_torch_stft = bool(patch_torch_stft)
        # Retained as an API-compatible configuration value for callers that
        # record AudioCraft's epsilon in experiment metadata. The default score
        # path uses log_softmax directly and does not perturb probabilities.
        self.epsilon = float(epsilon)
        self.quiet_backend = bool(quiet_backend)

        for direction in ("pq", "qp", "both"):
            self.add_state(
                f"{direction}_sum",
                default=torch.tensor(0.0, dtype=torch.float64),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"{direction}_sum_sq",
                default=torch.tensor(0.0, dtype=torch.float64),
                dist_reduce_fx="sum",
            )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

        if not lazy_load and self.classifier is None:
            self._ensure_classifier()
        elif self.classifier is not None:
            self._prepare_classifier(self.classifier)

    @property
    def max_frames(self) -> int:
        return int(round(self.sample_rate * self.max_duration_seconds))

    def _prepare_classifier(self, classifier: Any) -> Any:
        if hasattr(classifier, "eval"):
            classifier.eval()
        if isinstance(classifier, nn.Module):
            classifier.requires_grad_(False)
            classifier.to(self.pq_sum.device)
        return classifier

    def _ensure_classifier(self) -> Any:
        # pairwise_scores() runs under inference_mode, but the lazily created
        # weights must remain normal tensors so the cached model can later move
        # from GPU to CPU and back across epoch callbacks.
        with torch.inference_mode(False):
            if self.classifier is None:
                loader = self.classifier_loader or _load_default_passt_classifier
                self.classifier = loader()
                if self.classifier is None:
                    raise RuntimeError("classifier_loader returned None.")
            return self._prepare_classifier(self.classifier)

    def _prepare_audio(self, audio: Tensor | Any, sample_rate: int) -> Tensor:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        waveform = to_mono_batch(audio)
        if waveform.shape[-1] == 0:
            raise ValueError("Audio clips must contain at least one sample.")
        if sample_rate != self.sample_rate:
            try:
                import torchaudio
            except ImportError as exc:
                raise ImportError(
                    "torchaudio is required for PaSST resampling."
                ) from exc
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.sample_rate
            )
        if waveform.shape[-1] > self.max_frames:
            raise ValueError(
                "PaSSTKLDivergence expects fixed clips no longer than "
                f"{self.max_duration_seconds:g} seconds at {self.sample_rate} Hz; "
                f"got {waveform.shape[-1]} frames."
            )
        return waveform

    def _classify(self, waveform: Tensor) -> Tensor:
        device = self.pq_sum.device
        waveform = waveform.to(device=device, dtype=torch.float32)
        with quiet_external_output(self.quiet_backend):
            classifier = self._ensure_classifier()
            with _legacy_stft_compatibility(self.patch_torch_stft):
                output = classifier(waveform)
        logits = _extract_logits(output).to(device=device)
        if logits.shape[0] != waveform.shape[0]:
            raise ValueError(
                "PaSST classifier changed the batch dimension: "
                f"expected {waveform.shape[0]}, got {logits.shape[0]}."
            )
        return logits

    @torch.inference_mode()
    def pairwise_scores(
        self,
        generated_audio: Tensor,
        reference_audio: Tensor,
        *,
        sample_rate: int | None = None,
    ) -> dict[str, Tensor]:
        """Return per-clip ``pq``, ``qp``, and ``both`` scores without updating."""

        source_rate = self.sample_rate if sample_rate is None else int(sample_rate)
        generated = self._prepare_audio(generated_audio, source_rate)
        reference = self._prepare_audio(reference_audio, source_rate)
        generated, reference = match_batch(generated, reference)

        generated_logits = self._classify(generated).double()
        reference_logits = self._classify(reference).double()
        if generated_logits.shape != reference_logits.shape:
            raise ValueError(
                "Generated and reference PaSST logits must have matching shapes; "
                f"got {tuple(generated_logits.shape)} and "
                f"{tuple(reference_logits.shape)}."
            )

        # Work in float64 and obtain log probabilities directly from logits.
        # This is the AudioCraft direction/normalization without perturbing tiny
        # AudioSet classes by adding epsilon to an already normalized vector.
        generated_log_probabilities = F.log_softmax(generated_logits, dim=-1)
        reference_log_probabilities = F.log_softmax(reference_logits, dim=-1)
        generated_probabilities = generated_log_probabilities.exp()
        reference_probabilities = reference_log_probabilities.exp()
        pq = (
            F.kl_div(
                generated_log_probabilities,
                reference_probabilities,
                reduction="none",
            )
            .sum(dim=-1)
            .clamp_min(0.0)
        )
        qp = (
            F.kl_div(
                reference_log_probabilities,
                generated_probabilities,
                reduction="none",
            )
            .sum(dim=-1)
            .clamp_min(0.0)
        )
        both = pq + qp
        return {"pq": pq, "qp": qp, "both": both}

    @torch.inference_mode()
    def per_sample(
        self,
        generated_audio: Tensor,
        reference_audio: Tensor,
        *,
        sample_rate: int | None = None,
    ) -> Tensor:
        """Return the primary ``KL(reference || generated)`` for every clip."""

        return self.pairwise_scores(
            generated_audio, reference_audio, sample_rate=sample_rate
        )["pq"]

    @torch.inference_mode()
    def update(
        self,
        generated_audio: Tensor,
        reference_audio: Tensor,
        *,
        sample_rate: int | None = None,
    ) -> None:
        """Accumulate paired per-clip PaSST KL scores."""

        scores = self.pairwise_scores(
            generated_audio, reference_audio, sample_rate=sample_rate
        )

        for direction, values in scores.items():
            values = values.to(device=self.pq_sum.device, dtype=torch.float64)
            getattr(self, f"{direction}_sum").add_(values.sum())
            getattr(self, f"{direction}_sum_sq").add_(values.square().sum())
        self.total += torch.tensor(
            scores["pq"].numel(), device=self.total.device, dtype=self.total.dtype
        )

    def _mean_std(self, direction: str) -> tuple[Tensor, Tensor]:
        if int(self.total.item()) == 0:
            nan = self.pq_sum.new_tensor(float("nan"))
            return nan, nan.clone()
        count = self.total.to(dtype=torch.float64)
        mean = getattr(self, f"{direction}_sum") / count
        second_moment = getattr(self, f"{direction}_sum_sq") / count
        # Population standard deviation (correction=0), as requested for the
        # fixed evaluation cohort aggregated by this metric.
        std = (second_moment - mean.square()).clamp_min(0.0).sqrt()
        return mean, std

    def compute(self) -> dict[str, Tensor]:
        """Return directional KL means and population standard deviations."""

        pq, pq_std = self._mean_std("pq")
        qp, qp_std = self._mean_std("qp")
        both, both_std = self._mean_std("both")
        return {
            "kld": pq,
            "kld_std": pq_std,
            "kld_pq": pq,
            "kld_pq_std": pq_std,
            "kld_qp": qp,
            "kld_qp_std": qp_std,
            "kld_both": both,
            "kld_both_std": both_std,
        }


__all__ = ["PaSSTKLDivergence"]
