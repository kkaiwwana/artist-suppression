"""A Lightning-free MusicGen module based on Hugging Face Transformers.

The module intentionally owns only model computation.  It does not import or
depend on AudioCraft, PyTorch Lightning, a Dataset, or a Trainer.  The outer
project can therefore call ``training_step`` and ``validation_step`` from its
own Lightning module, or call ``forward`` directly in an ordinary PyTorch
loop.

Batch contract
--------------
The recommended dataset batch is a mapping with these fields:

``input_ids``
    Text tokenizer ids, shape ``[batch, text_length]``.
``attention_mask``
    Text padding mask, shape ``[batch, text_length]``.
``audio_tokens``
    EnCodec codes, shape ``[batch, num_codebooks, audio_length]``.  Values are
    integer codebook ids.  ``decoder_input_ids`` may be used instead, with the
    same shape.  The wrapper flattens this to the Hugging Face convention of
    ``[batch * num_codebooks, audio_length]`` for explicit decoder inputs.
    During training, the wrapper constructs the same right-shifted
    teacher-forcing inputs as Transformers and computes padding-safe loss
    outside the backbone.
``labels`` (optional)
    Target EnCodec codes in the same ``[batch, num_codebooks, audio_length]``
    layout.  If omitted by ``training_step``, ``audio_tokens`` are used as
    labels and padding-safe MusicGen cross-entropy is computed by this wrapper.
``decoder_attention_mask`` (optional)
    Audio-token mask, shape ``[batch, audio_length]``.

The text input can also be supplied as precomputed ``encoder_outputs`` when a
caller is training the standalone MusicGen decoder.  ``text_input_ids`` and
``text_attention_mask`` are accepted aliases for the first two fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.module.lora_adapter import (
    LoRALinear,
    freeze_module_except_lora,
    inject_attention_lora,
)


log = logging.getLogger(__name__)


@dataclass
class MusicGenOutput:
    """Stable, backend-independent output for forward and step methods."""

    loss: Optional[Tensor]
    logits: Tensor
    labels: Optional[Tensor]
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    raw_output: Any = None


@dataclass
class GenerationOutput:
    """Output returned by :meth:`MusicGen.generate` when ``return_dict=True``."""

    audio_values: Tensor
    raw_output: Any = None


@dataclass(frozen=True)
class HiddenStateHookContext:
    """Metadata passed to a hidden-state capture/intervention callback."""

    name: str
    block_index: int
    module: nn.Module


HiddenStateCallback = Callable[[Tensor, HiddenStateHookContext], Optional[Tensor]]


class MusicGen(nn.Module):
    """Pure PyTorch wrapper around Transformers MusicGen.

    Construct the wrapper around an already-created model with ``backbone`` or
    use :meth:`from_pretrained` for the normal text-to-music workflow. When
    ``use_adapter=True``, standard linear LoRA branches are inserted into the
    selected projections of every decoder attention module. Unless explicitly
    overridden, this also freezes the original model parameters.

    Args:
        backbone: A ``MusicgenForConditionalGeneration`` instance.  Passing a
            model object keeps the class usable with locally constructed tiny
            configs in tests and notebooks.
        processor: An ``AutoProcessor`` instance used by ``generate``.
        use_adapter: Insert standard LoRA into decoder attention projections.
        freeze_backbone: Freeze the base model.  Defaults to ``use_adapter``.
        adapter_rank: Bottleneck rank for each adapter.
        adapter_alpha: LoRA alpha.  Defaults to the rank.
        adapter_targets: Attention projections to adapt. Defaults to Q and V.
        adapter_dropout: Dropout before the adapter down projection.
        adapter_zero_init: Zero-initialise LoRA B for an identity start.
    """

    # Dataset/control metadata must remain available to the outer Lightning
    # module but must not be forwarded as unexpected Hugging Face arguments.
    NON_MODEL_BATCH_KEYS = {
        "text",
        "texts",
        "prompt",
        "prompts",
        "description",
        "descriptions",
        "metadata",
        "artist_label",
        "genre_labels",
        "concept_ids",
        "concept_weights",
        "concept_targets",
        "control_direction",
        "suppressed_artist_ids",
        "generation_inputs",
        "generation_kwargs",
    }

    def __init__(
        self,
        backbone: nn.Module,
        processor: Optional[Any] = None,
        *,
        use_adapter: bool = False,
        freeze_backbone: Optional[bool] = None,
        adapter_rank: int = 8,
        adapter_alpha: Optional[float] = None,
        adapter_targets: Sequence[str] = ("q_proj", "v_proj"),
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
        adapter_checkpoint: Optional[str | Path] = None,
        adapter_checkpoint_strict: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")

        self.backbone = backbone
        self._patch_legacy_decoder_start_token(backbone)
        self.processor = processor
        self.use_adapter = bool(use_adapter)
        self.adapter_module_names: list[str] = []
        self.loaded_adapter_checkpoint: Optional[str] = None
        self._hidden_state_handles: List[Any] = []

        if adapter_checkpoint is not None and not self.use_adapter:
            raise ValueError(
                "adapter_checkpoint requires use_adapter=True so matching adapter "
                "modules exist before weights are loaded"
            )

        if self.use_adapter:
            adapter_root = self._find_decoder_root(backbone)
            self.adapter_module_names = inject_attention_lora(
                adapter_root,
                rank=adapter_rank,
                alpha=adapter_alpha,
                targets=adapter_targets,
                dropout=adapter_dropout,
                zero_init=adapter_zero_init,
            )
            if not self.adapter_module_names:
                raise RuntimeError(
                    "use_adapter=True but no decoder attention modules were found"
                )
            if adapter_checkpoint is not None:
                self.load_adapter_checkpoint(
                    adapter_checkpoint,
                    strict=adapter_checkpoint_strict,
                )

        if freeze_backbone is None:
            freeze_backbone = self.use_adapter
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            freeze_module_except_lora(self.backbone)

        self.num_codebooks = self._read_num_codebooks(backbone)
        self.pad_token_id = self._read_config_value(
            "pad_token_id", config_path=("decoder",)
        )
        if self.pad_token_id is None:
            self.pad_token_id = self._read_config_value("pad_token_id")
        self.decoder_start_token_id = self._read_decoder_start_token_id()
        self.audio_sample_rate = self._read_config_value(
            "sampling_rate", config_path=("audio_encoder",)
        ) or 32000

    @staticmethod
    def _patch_legacy_decoder_start_token(backbone: nn.Module) -> None:
        """Make pre-Transformers-5 MusicGen configs trainable with labels.

        Older official checkpoints leave ``decoder_start_token_id`` unset but
        use the shared BOS/PAD special token (2048). Transformers 5 requires
        the start ID explicitly when it shifts labels for teacher forcing.
        """

        config = getattr(backbone, "config", None)
        decoder_config = getattr(config, "decoder", None)
        if decoder_config is None:
            return
        if getattr(decoder_config, "decoder_start_token_id", None) is not None:
            return
        start_token_id = getattr(decoder_config, "bos_token_id", None)
        if start_token_id is None:
            start_token_id = getattr(decoder_config, "pad_token_id", None)
        if start_token_id is None:
            raise ValueError(
                "MusicGen decoder config needs decoder_start_token_id, "
                "bos_token_id, or pad_token_id for teacher forcing"
            )
        decoder_config.decoder_start_token_id = int(start_token_id)

    @property
    def model(self) -> nn.Module:
        """Alias for callers used to accessing a Transformers model as ``.model``."""

        return self.backbone

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "facebook/musicgen-small",
        *,
        processor: Optional[Any] = None,
        processor_name_or_path: Optional[str] = None,
        use_adapter: bool = False,
        freeze_backbone: Optional[bool] = None,
        adapter_rank: int = 8,
        adapter_alpha: Optional[float] = None,
        adapter_targets: Sequence[str] = ("q_proj", "v_proj"),
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
        adapter_checkpoint: Optional[str | Path] = None,
        adapter_checkpoint_strict: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device | str] = None,
    ) -> "MusicGen":
        """Load an official or locally downloaded Transformers checkpoint.

        Transformers is imported here rather than at module import time so
        that adapter utilities and type checking remain usable in a minimal
        PyTorch environment.  ``model_kwargs`` is passed directly to
        ``MusicgenForConditionalGeneration.from_pretrained`` and can contain
        such options as ``torch_dtype``, ``cache_dir`` and
        ``local_files_only``.
        """

        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Transformers is required to load MusicGen. Install it with "
                "`pip install transformers` before calling from_pretrained()."
            ) from exc

        load_kwargs = dict(model_kwargs or {})
        backbone = MusicgenForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        if processor is None:
            processor_load_kwargs = dict(processor_kwargs or {})
            if "local_files_only" in load_kwargs and "local_files_only" not in processor_load_kwargs:
                processor_load_kwargs["local_files_only"] = load_kwargs["local_files_only"]
            processor = AutoProcessor.from_pretrained(
                processor_name_or_path or model_name_or_path,
                **processor_load_kwargs,
            )

        wrapped = cls(
            backbone,
            processor,
            use_adapter=use_adapter,
            freeze_backbone=freeze_backbone,
            adapter_rank=adapter_rank,
            adapter_alpha=adapter_alpha,
            adapter_targets=adapter_targets,
            adapter_dropout=adapter_dropout,
            adapter_zero_init=adapter_zero_init,
            adapter_checkpoint=adapter_checkpoint,
            adapter_checkpoint_strict=adapter_checkpoint_strict,
        )
        if device is not None:
            wrapped.to(device)
        return wrapped

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Any:
        """Run MusicGen's own text encoder for callers that cache text states.

        The returned object is accepted directly as the composite model's
        ``encoder_outputs`` and exposes ``last_hidden_state``.
        """

        encoder = getattr(self.backbone, "text_encoder", None)
        if encoder is None:
            get_encoder = getattr(self.backbone, "get_encoder", None)
            encoder = get_encoder() if callable(get_encoder) else None
        if not isinstance(encoder, nn.Module):
            raise RuntimeError("the MusicGen backbone does not expose a text encoder")
        kwargs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        try:
            return encoder(**kwargs, return_dict=True)
        except TypeError as error:
            if "return_dict" not in str(error):
                raise
            return encoder(**kwargs)

    @torch.no_grad()
    def decode_audio_tokens(
        self,
        audio_tokens: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Decode padded ``[B, Q, T]`` EnCodec IDs into reference audio.

        Samples are decoded independently so padding ID 2048 is never passed
        to EnCodec.  The returned audio is zero-padded to ``[B, C, S_max]``;
        the second tensor stores each decoded sample length.
        """

        if audio_tokens.ndim != 3:
            raise ValueError("audio_tokens must have shape [B, Q, T]")
        audio_encoder = getattr(self.backbone, "audio_encoder", None)
        if not isinstance(audio_encoder, nn.Module) or not hasattr(
            audio_encoder, "decode"
        ):
            raise RuntimeError("the MusicGen backbone has no EnCodec decoder")
        if attention_mask is not None and attention_mask.shape != (
            audio_tokens.shape[0],
            audio_tokens.shape[-1],
        ):
            raise ValueError("attention_mask must have shape [B, T]")

        decoded_samples: list[Tensor] = []
        lengths: list[int] = []
        for sample_index in range(audio_tokens.shape[0]):
            token_length = (
                int(attention_mask[sample_index].sum().item())
                if attention_mask is not None
                else audio_tokens.shape[-1]
            )
            if token_length <= 0:
                raise ValueError("every audio sample must contain at least one token")
            codes = audio_tokens[
                sample_index : sample_index + 1, :, :token_length
            ].long()
            decoded = audio_encoder.decode(
                audio_codes=codes.unsqueeze(0),
                audio_scales=[None],
                return_dict=True,
            )
            audio_values = getattr(decoded, "audio_values", None)
            if audio_values is None and isinstance(decoded, (tuple, list)):
                audio_values = decoded[0]
            if not isinstance(audio_values, Tensor) or audio_values.ndim != 3:
                raise RuntimeError("EnCodec decode must return audio shaped [B, C, S]")
            sample = audio_values[0]
            decoded_samples.append(sample)
            lengths.append(sample.shape[-1])

        max_length = max(lengths)
        channels = decoded_samples[0].shape[0]
        padded = decoded_samples[0].new_zeros(
            (len(decoded_samples), channels, max_length)
        )
        for index, sample in enumerate(decoded_samples):
            padded[index, :, : sample.shape[-1]] = sample
        return padded, torch.tensor(lengths, device=padded.device, dtype=torch.long)

    @staticmethod
    def _find_decoder_root(backbone: nn.Module) -> nn.Module:
        """Select the decoder subtree and avoid adapting the T5 text encoder."""

        decoder = getattr(backbone, "decoder", None)
        if isinstance(decoder, nn.Module):
            return decoder
        return backbone

    def _read_num_codebooks(self, backbone: nn.Module) -> Optional[int]:
        config = getattr(backbone, "config", None)
        decoder_config = getattr(config, "decoder", None)
        for candidate in (
            getattr(decoder_config, "num_codebooks", None),
            getattr(config, "num_codebooks", None),
            getattr(backbone, "num_codebooks", None),
        ):
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        return None

    def _read_config_value(
        self,
        name: str,
        *,
        config_path: Tuple[str, ...] = (),
    ) -> Optional[int]:
        config = getattr(self.backbone, "config", None)
        for path_item in config_path:
            config = getattr(config, path_item, None)
        value = getattr(config, name, None) if config is not None else None
        return value if isinstance(value, int) else None

    def _read_decoder_start_token_id(self) -> int:
        """Resolve the special token used to start teacher forcing."""

        for name in ("decoder_start_token_id", "bos_token_id", "pad_token_id"):
            for path in (("decoder",), ()):
                value = self._read_config_value(name, config_path=path)
                if value is not None:
                    return value
        if self.pad_token_id is not None:
            return self.pad_token_id
        raise ValueError("MusicGen needs a decoder start/BOS/PAD token ID")

    def _shift_audio_labels(self, labels: Tensor) -> Tensor:
        """Create MusicGen teacher-forcing inputs without invoking HF loss."""

        pad_token_id = self.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.decoder_start_token_id
        shifted = labels.new_full(labels.shape, int(pad_token_id))
        shifted[..., 0] = self.decoder_start_token_id
        if labels.shape[-1] > 1:
            shifted[..., 1:] = labels[..., :-1]
        shifted.masked_fill_(shifted.eq(-100), int(pad_token_id))
        return shifted

    @property
    def trainable_parameters(self):
        """Iterator over parameters visible to an outer optimizer."""

        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters)

    def _named_adapter_parameters(self) -> Dict[str, nn.Parameter]:
        """Return only trainable LoRA A/B parameters, never base weights."""

        parameters: Dict[str, nn.Parameter] = {}
        for module_name, module in self.named_modules():
            if not module_name or not isinstance(module, LoRALinear):
                continue
            parameters[f"{module_name}.lora_A.weight"] = module.lora_A.weight
            parameters[f"{module_name}.lora_B.weight"] = module.lora_B.weight
        return parameters

    def adapter_state_dict(self) -> Dict[str, Tensor]:
        """Return a compact CPU copy containing only adapter parameters."""

        parameters = self._named_adapter_parameters()
        if not parameters:
            raise RuntimeError("this MusicGen wrapper has no injected adapters")
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in parameters.items()
        }

    @staticmethod
    def _checkpoint_tensor_mapping(checkpoint_path: Path) -> Mapping[str, Tensor]:
        """Read a raw, compact, or Lightning checkpoint without executing code."""

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"adapter checkpoint must contain a mapping, got "
                f"{type(payload).__name__}"
            )
        for container_name in ("adapter_state_dict", "state_dict"):
            nested = payload.get(container_name)
            if isinstance(nested, Mapping):
                payload = nested
                break
        tensors = {
            str(name): value
            for name, value in payload.items()
            if isinstance(value, Tensor)
        }
        if not tensors:
            raise ValueError(
                f"adapter checkpoint contains no tensor state: {checkpoint_path}"
            )
        return tensors

    def load_adapter_checkpoint(
        self,
        checkpoint: str | Path,
        *,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load adapter parameters from an adapter-only or Lightning checkpoint.

        Stage-one Lightning checkpoints prefix generator keys with ``model.``.
        This loader matches each adapter parameter by its complete MusicGen key
        or by a unique suffix, so the same file can initialize a stage-two
        :class:`UnlearnableGenerationModel` without restoring trainer state,
        optimizer state, global step, or unrelated frozen MusicGen tensors.
        """

        if not self.use_adapter:
            raise RuntimeError("load_adapter_checkpoint requires use_adapter=True")
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"adapter checkpoint not found: {path}")

        expected = self._named_adapter_parameters()
        if not expected:
            raise RuntimeError("no adapter parameters are available to load")
        source = self._checkpoint_tensor_mapping(path)
        matched_source_keys: set[str] = set()
        missing: list[str] = []
        loaded: list[str] = []

        with torch.no_grad():
            for expected_name, parameter in expected.items():
                candidates = [
                    source_name
                    for source_name in source
                    if source_name == expected_name
                    or source_name.endswith(f".{expected_name}")
                ]
                if not candidates:
                    missing.append(expected_name)
                    continue
                if len(candidates) > 1:
                    raise RuntimeError(
                        f"adapter parameter {expected_name!r} has ambiguous "
                        f"checkpoint matches: {candidates}"
                    )
                source_name = candidates[0]
                value = source[source_name]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"adapter shape mismatch for {expected_name}: model expects "
                        f"{tuple(parameter.shape)}, checkpoint has {tuple(value.shape)}. "
                        "Use the same adapter_rank as stage one."
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                matched_source_keys.add(source_name)
                loaded.append(expected_name)

        legacy_nonlinear = sorted(
            name
            for name in source
            if ".adapter.down." in name or ".adapter.up." in name
        )
        if legacy_nonlinear:
            raise RuntimeError(
                "checkpoint contains the retired nonlinear output adapter and "
                "cannot initialize standard projection LoRA; retrain stage one"
            )
        unexpected = sorted(
            name
            for name in source
            if (".lora_A." in name or ".lora_B." in name)
            and name not in matched_source_keys
        )
        if strict and (missing or unexpected):
            raise RuntimeError(
                "adapter checkpoint did not exactly match the initialized adapters; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if not loaded:
            raise RuntimeError(
                f"no adapter parameters from {path} matched this MusicGen model"
            )
        self.loaded_adapter_checkpoint = str(path)
        log.info("Loaded %d adapter tensors from %s", len(loaded), path)
        return {
            "path": str(path),
            "loaded": loaded,
            "missing": missing,
            "unexpected": unexpected,
        }

    @staticmethod
    def _is_hidden_state_block(module: nn.Module) -> bool:
        """Identify decoder blocks without depending on a Transformers class."""

        class_name = module.__class__.__name__.lower()
        if "decoderlayer" in class_name or "transformerblock" in class_name:
            return True
        return hasattr(module, "self_attn") and hasattr(module, "final_layer_norm")

    @staticmethod
    def _parsed_block_index(name: str) -> Optional[int]:
        match = re.search(r"(?:layers|blocks|decoder_layers)\.(\d+)$", name)
        return int(match.group(1)) if match else None

    def _hidden_state_module_entries(self) -> List[Tuple[str, nn.Module, int]]:
        """Return decoder-layer modules and their stable ConceptLearner index."""

        root = self._find_decoder_root(self.backbone)
        entries: List[Tuple[str, nn.Module, int]] = []
        candidates = [
            (name, module)
            for name, module in root.named_modules()
            if name and self._is_hidden_state_block(module)
        ]
        for ordinal, (name, module) in enumerate(candidates):
            parsed_index = self._parsed_block_index(name)
            entries.append(
                (name, module, ordinal if parsed_index is None else parsed_index)
            )
        return entries

    def named_hidden_state_modules(self) -> Dict[str, nn.Module]:
        """List decoder-layer targets available to hidden-state hooks.

        Names are relative to the selected decoder root.  On a standard
        Transformers MusicGen model they look like
        ``model.decoder.layers.0``.  The names can be passed to
        :meth:`register_hidden_state_hook`.
        """

        return {name: module for name, module, _ in self._hidden_state_module_entries()}

    def _resolve_hidden_state_targets(
        self,
        module_names: Optional[str | int | Sequence[str | int]],
    ) -> List[Tuple[str, nn.Module, int]]:
        entries = self._hidden_state_module_entries()
        if not entries:
            raise RuntimeError(
                "no decoder-layer modules were found for hidden-state hooks"
            )
        by_name = {name: (name, module, index) for name, module, index in entries}
        by_index = {index: (name, module, index) for name, module, index in entries}
        if module_names is None:
            return entries
        requested = [module_names] if isinstance(module_names, (str, int)) else list(module_names)
        targets: List[Tuple[str, nn.Module, int]] = []
        for requested_name in requested:
            if isinstance(requested_name, int):
                if requested_name not in by_index:
                    raise IndexError(
                        f"hidden-state block index {requested_name} is unavailable; "
                        f"available indices are {sorted(by_index)}"
                    )
                targets.append(by_index[requested_name])
            else:
                if requested_name not in by_name:
                    raise KeyError(
                        f"unknown hidden-state module {requested_name!r}; "
                        f"available modules are {list(by_name)}"
                    )
                targets.append(by_name[requested_name])
        return targets

    @staticmethod
    def _extract_hidden_state(output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            return output[0]
        hidden = getattr(output, "last_hidden_state", None)
        if isinstance(hidden, Tensor):
            return hidden
        raise TypeError(
            "a hooked decoder layer must return a Tensor, a tuple/list whose "
            "first item is a Tensor, or an object with last_hidden_state"
        )

    @staticmethod
    def _replace_hidden_state(output: Any, hidden_state: Tensor) -> Any:
        if isinstance(output, Tensor):
            return hidden_state
        if isinstance(output, tuple):
            return (hidden_state,) + output[1:]
        if isinstance(output, list):
            return [hidden_state, *output[1:]]
        if hasattr(output, "last_hidden_state"):
            # HF layer outputs are normally tuples.  This fallback supports
            # custom ModelOutput-like objects without coupling this module to
            # Transformers' ModelOutput implementation.
            import copy

            copied = copy.copy(output)
            copied.last_hidden_state = hidden_state
            return copied
        raise TypeError(f"unsupported decoder-layer output type {type(output).__name__}")

    def register_hidden_state_hook(
        self,
        callback: HiddenStateCallback,
        module_names: Optional[str | int | Sequence[str | int]] = None,
        *,
        prepend: bool = False,
    ) -> List[Any]:
        """Register capture/intervention callbacks on decoder layer outputs.

        ``callback`` receives ``(hidden_state, context)`` after a decoder layer
        has run.  Returning ``None`` keeps the original hidden state; returning
        a Tensor replaces it and that replacement continues through the rest
        of the MusicGen forward pass.  This makes both read-only capture and
        ConceptLearner-style hidden-state intervention possible without
        modifying the Transformers implementation.

        The returned handles support ``handle.remove()``.  The model also
        provides :meth:`clear_hidden_state_hooks` for bulk cleanup.
        """

        if not callable(callback):
            raise TypeError("callback must be callable")
        handles: List[Any] = []
        for name, module, block_index in self._resolve_hidden_state_targets(module_names):
            context = HiddenStateHookContext(
                name=name,
                block_index=block_index,
                module=module,
            )

            def _hook(_module, _inputs, output, *, _context=context):
                hidden_state = self._extract_hidden_state(output)
                replacement = callback(hidden_state, _context)
                if replacement is None:
                    return output
                if not isinstance(replacement, Tensor):
                    raise TypeError("hidden-state callback must return Tensor or None")
                if replacement.shape != hidden_state.shape:
                    raise ValueError(
                        f"hidden-state callback changed shape from "
                        f"{tuple(hidden_state.shape)} to {tuple(replacement.shape)}"
                    )
                return self._replace_hidden_state(output, replacement)

            try:
                handle = module.register_forward_hook(_hook, prepend=prepend)
            except TypeError:  # compatibility with older PyTorch versions
                handle = module.register_forward_hook(_hook)
            handles.append(handle)
            self._hidden_state_handles.append(handle)
        return handles

    # This alias makes the intent clearer at call sites that modify rather
    # than merely inspect layer outputs.
    register_hidden_state_intervention = register_hidden_state_hook

    def remove_hidden_state_hooks(self, handles: Sequence[Any]) -> None:
        """Remove selected handles and forget them from the wrapper registry."""

        for handle in handles:
            handle.remove()
            if handle in self._hidden_state_handles:
                self._hidden_state_handles.remove(handle)

    def clear_hidden_state_hooks(self) -> None:
        """Remove all hooks registered through this wrapper."""

        for handle in self._hidden_state_handles:
            handle.remove()
        self._hidden_state_handles.clear()

    @contextmanager
    def capture_hidden_states(
        self,
        module_names: Optional[str | int | Sequence[str | int]] = None,
        *,
        detach: bool = True,
        clone: bool = False,
    ) -> Iterator[Dict[str, List[Tensor]]]:
        """Temporarily capture decoder hidden states keyed by module name.

        Each value is a list because generation or gradient checkpointing can
        execute a layer more than once.  Set ``detach=False`` when the capture
        itself must remain part of the autograd graph.
        """

        captured: Dict[str, List[Tensor]] = {}

        def _capture(hidden_state: Tensor, context: HiddenStateHookContext) -> None:
            value = hidden_state.detach() if detach else hidden_state
            if clone:
                value = value.clone()
            captured.setdefault(context.name, []).append(value)
            return None

        handles = self.register_hidden_state_hook(_capture, module_names)
        try:
            yield captured
        finally:
            self.remove_hidden_state_hooks(handles)

    @contextmanager
    def intervene_hidden_states(
        self,
        callback: HiddenStateCallback,
        module_names: Optional[str | int | Sequence[str | int]] = None,
        *,
        prepend: bool = False,
    ) -> Iterator[List[Any]]:
        """Temporarily install a hidden-state intervention callback."""

        handles = self.register_hidden_state_hook(
            callback,
            module_names,
            prepend=prepend,
        )
        try:
            yield handles
        finally:
            self.remove_hidden_state_hooks(handles)

    def register_concept_learner_hooks(
        self,
        concept_learner: nn.Module,
        condition: Optional[Any] = None,
        module_names: Optional[str | int | Sequence[str | int]] = None,
        *,
        condition_kwargs: Optional[Dict[str, Any]] = None,
        intervention_scale: float | Tensor = 1.0,
        batch_repeat_interleave: Optional[int] = None,
        cfg_conditional_only: bool = False,
        details_callback: Optional[
            Callable[[Tensor, Mapping[str, Tensor], HiddenStateHookContext], None]
        ] = None,
        prepend: bool = False,
    ) -> List[Any]:
        """Connect ``src.model.module.ConceptLearner`` to decoder layers.

        ``condition`` should normally be prepared once with
        ``concept_learner.prepare_condition(...)`` and reused across layers.
        Alternatively pass ``condition_kwargs`` and this method prepares it.
        The learner must have a ``forward(hidden_state, block_index, condition,
        intervention_scale=...)`` interface, which is the interface provided by
        the project's existing ``ConceptLearner``.
        """

        if condition is None:
            if condition_kwargs is None:
                raise ValueError("provide condition or condition_kwargs")
            prepare_condition = getattr(concept_learner, "prepare_condition", None)
            if prepare_condition is None:
                raise TypeError(
                    "condition_kwargs requires a concept learner with "
                    "prepare_condition"
                )
            condition = prepare_condition(**condition_kwargs)

        if batch_repeat_interleave is None:
            batch_repeat_interleave = int(self.num_codebooks or 1)

        def _apply(hidden_state: Tensor, context: HiddenStateHookContext) -> Tensor:
            result = concept_learner(
                hidden_state,
                block_index=context.block_index,
                condition=condition,
                intervention_scale=intervention_scale,
                batch_repeat_interleave=batch_repeat_interleave,
                cfg_conditional_only=cfg_conditional_only,
                return_details=details_callback is not None,
            )
            if details_callback is None:
                return result
            if not isinstance(result, Mapping):
                raise TypeError(
                    "concept learner must return a details mapping when "
                    "details_callback is supplied"
                )
            details_callback(hidden_state, result, context)
            changed = result.get("hidden_state")
            if not isinstance(changed, Tensor):
                raise TypeError("concept learner details must contain hidden_state")
            return changed

        return self.register_hidden_state_hook(
            _apply,
            module_names,
            prepend=prepend,
        )

    @staticmethod
    def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    def _prepare_batch(
        self,
        batch: Mapping[str, Any],
        *,
        use_audio_as_labels: bool,
    ) -> Tuple[Dict[str, Any], Optional[Tensor], Optional[Tuple[int, int, int]]]:
        """Convert the project batch contract to HF's MusicGen contract."""

        values = dict(batch)
        input_ids = self._first(values, "input_ids", "text_input_ids")
        attention_mask = self._first(values, "attention_mask", "text_attention_mask")
        explicit_decoder_input_ids = values.get("decoder_input_ids")
        decoder_input_ids = self._first(values, "decoder_input_ids", "audio_tokens")
        labels = self._first(values, "labels", "audio_labels", "target_audio_tokens")
        if labels is None and use_audio_as_labels:
            labels = decoder_input_ids
            # Hugging Face MusicGen applies its codebook delay/teacher-forcing
            # shift only when decoder_input_ids are omitted. ``audio_tokens``
            # is the project's target shortcut; an explicitly supplied
            # decoder_input_ids tensor remains caller-controlled.
            if explicit_decoder_input_ids is None:
                decoder_input_ids = None

        standard_labels: Optional[Tensor] = None
        shape_info: Optional[Tuple[int, int, int]] = None
        if isinstance(labels, Tensor):
            standard_labels, shape_info = self._standardize_audio_layout(
                labels, name="labels"
            )
        elif labels is not None:
            labels_tensor = torch.as_tensor(labels)
            standard_labels, shape_info = self._standardize_audio_layout(
                labels_tensor, name="labels"
            )

        # Transformers computes its own loss whenever ``labels`` are passed.
        # On Windows ROCm, the first right-padded batch can hang in the
        # internal cross-entropy ignore-index kernel. Reproduce HF's
        # shift_tokens_right here and let this wrapper compute loss only over
        # valid targets instead.
        labels_only_teacher_forcing = (
            standard_labels is not None and decoder_input_ids is None
        )
        if labels_only_teacher_forcing:
            decoder_input_ids = self._shift_audio_labels(standard_labels)

        if decoder_input_ids is not None:
            decoder_input_ids = torch.as_tensor(decoder_input_ids)
            if decoder_input_ids.ndim == 3:
                batch_size, num_codebooks, sequence_length = decoder_input_ids.shape
                if self.num_codebooks is not None and num_codebooks != self.num_codebooks:
                    raise ValueError(
                        f"expected {self.num_codebooks} codebooks, got {num_codebooks}"
                    )
                shape_info = shape_info or (
                    batch_size,
                    num_codebooks,
                    sequence_length,
                )
                decoder_input_ids = decoder_input_ids.reshape(
                    batch_size * num_codebooks, sequence_length
                )
            elif decoder_input_ids.ndim != 2:
                raise ValueError(
                    "decoder_input_ids/audio_tokens must have shape [B, Q, T] "
                    f"or [B*Q, T], got {tuple(decoder_input_ids.shape)}"
                )

        payload: Dict[str, Any] = {
            key: value
            for key, value in values.items()
            if key
            not in self.NON_MODEL_BATCH_KEYS
            and key
            not in {
                "audio_tokens",
                "audio_labels",
                "target_audio_tokens",
                "text_input_ids",
                "text_attention_mask",
                "labels",
                "decoder_input_ids",
                "batch_idx",
                "batch_size",
                "labels_layout",
            }
        }
        if input_ids is not None:
            payload["input_ids"] = input_ids
        if attention_mask is not None:
            payload["attention_mask"] = attention_mask
        if decoder_input_ids is not None:
            payload["decoder_input_ids"] = decoder_input_ids

        # Cached targets are right-padded. Valid causal positions cannot see
        # that future padding, and padded targets are excluded by the wrapper
        # loss. Explicit decoder prompts still retain their caller mask.
        if labels_only_teacher_forcing:
            payload.pop("decoder_attention_mask", None)

        # A decoder-only MusicgenForCausalLM uses input_ids rather than the
        # composite model's decoder_input_ids.  This is useful when training
        # on cached text encoder states and keeps this wrapper self-contained.
        if self._is_decoder_only():
            if "decoder_input_ids" in payload:
                payload["input_ids"] = payload.pop("decoder_input_ids")
            if "decoder_attention_mask" in payload and "attention_mask" not in payload:
                payload["attention_mask"] = payload.pop("decoder_attention_mask")
            if "encoder_outputs" in payload and "encoder_hidden_states" not in payload:
                encoder_outputs = payload.pop("encoder_outputs")
                if isinstance(encoder_outputs, (tuple, list)):
                    payload["encoder_hidden_states"] = encoder_outputs[0]
                else:
                    payload["encoder_hidden_states"] = getattr(
                        encoder_outputs,
                        "last_hidden_state",
                        encoder_outputs,
                    )

        # ``None`` values are rejected by some Transformers versions.
        payload = {key: value for key, value in payload.items() if value is not None}
        return payload, standard_labels, shape_info

    def _standardize_audio_layout(
        self, labels: Tensor, *, name: str
    ) -> Tuple[Tensor, Tuple[int, int, int]]:
        if labels.ndim == 3:
            batch_size, num_codebooks, sequence_length = labels.shape
            if self.num_codebooks is not None and num_codebooks != self.num_codebooks:
                raise ValueError(
                    f"{name} must have Q={self.num_codebooks} at dimension 1, "
                    f"got shape {tuple(labels.shape)}"
                )
            return labels.long(), (batch_size, num_codebooks, sequence_length)
        if labels.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, Q, T] or [B*Q, T], "
                f"got {tuple(labels.shape)}"
            )
        if self.num_codebooks is None:
            raise ValueError(
                "a flattened audio tensor requires the model's num_codebooks "
                "configuration"
            )
        flattened_batch, sequence_length = labels.shape
        if flattened_batch % self.num_codebooks:
            raise ValueError(
                f"flattened {name} first dimension {flattened_batch} is not divisible "
                f"by Q={self.num_codebooks}"
            )
        batch_size = flattened_batch // self.num_codebooks
        return labels.reshape(batch_size, self.num_codebooks, sequence_length).long(), (
            batch_size,
            self.num_codebooks,
            sequence_length,
        )

    def _is_decoder_only(self) -> bool:
        class_name = self.backbone.__class__.__name__.lower()
        return "causallm" in class_name and not hasattr(self.backbone, "text_encoder")

    def _reshape_logits(
        self,
        logits: Tensor,
        shape_info: Optional[Tuple[int, int, int]],
    ) -> Tensor:
        if logits.ndim == 4:
            if logits.shape[1] == (shape_info[1] if shape_info else logits.shape[1]):
                return logits
            if shape_info and logits.shape[-2] == shape_info[1]:
                return logits.transpose(1, 2)
            return logits
        if logits.ndim != 3:
            raise ValueError(f"expected logits with 3 or 4 dimensions, got {logits.shape}")
        if shape_info is None:
            if self.num_codebooks is None or logits.shape[0] % self.num_codebooks:
                raise ValueError("cannot infer MusicGen codebook layout from logits")
            batch_size = logits.shape[0] // self.num_codebooks
            num_codebooks = self.num_codebooks
        else:
            batch_size, num_codebooks, _ = shape_info
        if logits.shape[0] == batch_size * num_codebooks:
            return logits.reshape(batch_size, num_codebooks, logits.shape[1], logits.shape[2])
        if logits.shape[0] == batch_size and num_codebooks == 1:
            return logits.unsqueeze(1)
        raise ValueError(
            f"cannot reshape logits {tuple(logits.shape)} using "
            f"batch/codebooks {(batch_size, num_codebooks)}"
        )

    def _compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute MusicGen's mean per-codebook cross-entropy when needed."""

        losses = []
        for codebook in range(logits.shape[1]):
            codebook_labels = labels[:, codebook]
            valid = codebook_labels.ne(-100)
            if self.pad_token_id is not None:
                valid &= codebook_labels.ne(self.pad_token_id)
            if valid.any():
                losses.append(
                    F.cross_entropy(
                        logits[:, codebook][valid],
                        codebook_labels[valid],
                    )
                )
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _compute_sample_losses(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return per-sample and per-token MusicGen cross-entropy.

        ``sample_losses`` has shape ``[B]`` and averages valid tokens inside
        each codebook before averaging valid codebooks. ``token_losses`` has
        shape ``[B, Q, T]`` and is zero at ignored/padding positions.  The
        unreduced values allow an outer module to apply different objectives
        to retain and suppression subsets without changing the base model's
        ordinary scalar loss contract.
        """

        labels = labels.to(device=logits.device)
        valid = labels.ne(-100)
        if self.pad_token_id is not None:
            valid &= labels.ne(self.pad_token_id)
        safe_labels = labels.masked_fill(~valid, 0)
        token_losses = F.cross_entropy(
            logits.permute(0, 3, 1, 2).float(),
            safe_labels,
            reduction="none",
        )
        token_losses = token_losses * valid.to(dtype=token_losses.dtype)

        valid_token_counts = valid.sum(dim=-1)
        per_codebook = token_losses.sum(dim=-1) / valid_token_counts.clamp_min(1)
        valid_codebooks = valid_token_counts.gt(0)
        sample_losses = (
            per_codebook.sum(dim=-1)
            / valid_codebooks.sum(dim=-1).clamp_min(1)
        )
        return sample_losses, token_losses

    def _forward_prepared(
        self,
        payload: Dict[str, Any],
        standard_labels: Optional[Tensor],
        shape_info: Optional[Tuple[int, int, int]],
    ) -> MusicGenOutput:
        try:
            raw_output = self.backbone(**payload, return_dict=True)
        except TypeError as exc:
            # This fallback makes the module friendly to tiny test doubles and
            # older Transformers versions that do not expose return_dict.
            if "return_dict" not in str(exc):
                raise
            raw_output = self.backbone(**payload)
        raw_logits = getattr(raw_output, "logits", None)
        raw_loss = getattr(raw_output, "loss", None)
        if raw_logits is None and isinstance(raw_output, (tuple, list)):
            # Transformers' tuple output is either (logits, ...) or, when
            # labels are supplied, (loss, logits, ...).
            if (
                len(raw_output) > 1
                and isinstance(raw_output[0], Tensor)
                and raw_output[0].ndim == 0
            ):
                raw_loss = raw_output[0]
                raw_logits = raw_output[1]
            else:
                raw_logits = raw_output[0]
        if raw_logits is None:
            raise RuntimeError("the MusicGen backbone did not return logits")
        logits = self._reshape_logits(raw_logits, shape_info)
        loss = raw_loss
        if loss is None and standard_labels is not None:
            loss = self._compute_loss(logits, standard_labels.to(logits.device))
        hidden_states = getattr(raw_output, "hidden_states", None)
        return MusicGenOutput(
            loss=loss,
            logits=logits,
            labels=standard_labels,
            hidden_states=hidden_states,
            raw_output=raw_output,
        )

    def forward(self, batch: Optional[Mapping[str, Any]] = None, **kwargs) -> MusicGenOutput:
        """Run the wrapped MusicGen model using the project batch contract."""

        values: Dict[str, Any] = dict(batch or {})
        values.update(kwargs)
        payload, labels, shape_info = self._prepare_batch(
            values,
            use_audio_as_labels=False,
        )
        return self._forward_prepared(payload, labels, shape_info)

    def _step_metrics(self, output: MusicGenOutput) -> Dict[str, Tensor]:
        metrics: Dict[str, Tensor] = {}
        if output.loss is not None:
            loss = output.loss.detach()
            metrics["loss"] = loss
            metrics["perplexity"] = loss.float().exp().clamp(max=1e12)
        if output.labels is not None:
            labels = output.labels.to(output.logits.device)
            valid = labels.ne(-100)
            if self.pad_token_id is not None:
                valid &= labels.ne(self.pad_token_id)
            predictions = output.logits.argmax(dim=-1)
            if valid.any():
                metrics["token_accuracy"] = (
                    predictions[valid] == labels[valid]
                ).float().mean()
            else:
                metrics["token_accuracy"] = output.logits.new_zeros(())
        return metrics

    def training_step(self, batch: Mapping[str, Any], batch_idx: int = 0) -> Dict[str, Any]:
        """Compute the original MusicGen LM loss and training diagnostics.

        The returned live ``loss`` tensor is intentionally suitable for
        ``outer_lightning_module.manual_backward(result["loss"])`` or a normal
        ``result["loss"].backward()`` call.  No optimizer, logging, or trainer
        state is touched here.
        """

        del batch_idx
        payload, labels, shape_info = self._prepare_batch(
            batch,
            use_audio_as_labels=True,
        )
        output = self._forward_prepared(payload, labels, shape_info)
        if output.loss is None:
            raise ValueError(
                "training_step requires labels or audio_tokens so that the "
                "MusicGen language-model loss can be computed"
            )
        sample_losses, token_losses = self._compute_sample_losses(
            output.logits,
            output.labels,
        )
        return {
            "loss": output.loss,
            "sample_losses": sample_losses,
            "token_losses": token_losses,
            "logits": output.logits,
            "labels": output.labels,
            "metrics": self._step_metrics(output),
            "output": output,
        }

    @torch.no_grad()
    def validation_step(
        self, batch: Mapping[str, Any], batch_idx: int = 0
    ) -> Dict[str, Any]:
        """Compute validation loss and diagnostics without changing trainer state."""

        del batch_idx
        payload, labels, shape_info = self._prepare_batch(
            batch,
            use_audio_as_labels=True,
        )
        output = self._forward_prepared(payload, labels, shape_info)
        if output.loss is None:
            raise ValueError(
                "validation_step requires labels or audio_tokens so that the "
                "MusicGen language-model loss can be computed"
            )
        sample_losses, token_losses = self._compute_sample_losses(
            output.logits,
            output.labels,
        )
        return {
            "loss": output.loss,
            "sample_losses": sample_losses,
            "token_losses": token_losses,
            "logits": output.logits,
            "labels": output.labels,
            "metrics": self._step_metrics(output),
            "output": output,
        }

    @torch.no_grad()
    def generate(
        self,
        text: Optional[str | Sequence[str]] = None,
        *,
        inputs: Optional[Mapping[str, Any]] = None,
        return_dict: bool = False,
        **generation_kwargs: Any,
    ) -> Tensor | GenerationOutput:
        """Generate waveform audio from text using the stored processor.

        The returned tensor has the Transformers MusicGen shape
        ``[batch, channels, samples]``.  ``inputs`` can be supplied when the
        caller already ran the processor.  All additional keyword arguments,
        such as ``do_sample``, ``guidance_scale`` and ``max_new_tokens``, are
        forwarded to ``backbone.generate``.
        """

        if inputs is None:
            if text is None:
                raise ValueError("provide text or preprocessed inputs")
            if self.processor is None:
                raise RuntimeError(
                    "generate(text=...) requires an AutoProcessor; pass one to "
                    "the constructor or load the model with from_pretrained()"
                )
            prompts = [text] if isinstance(text, str) else list(text)
            inputs = self.processor(
                text=prompts,
                padding=True,
                return_tensors="pt",
            )

        device = next(self.parameters()).device
        prepared_inputs = {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in dict(inputs).items()
        }
        raw_output = self.backbone.generate(**prepared_inputs, **generation_kwargs)
        audio_values = getattr(raw_output, "audio_values", None)
        if audio_values is None:
            audio_values = raw_output
        if not isinstance(audio_values, Tensor):
            raise TypeError(
                "MusicGen generate() must return a Tensor or an object with "
                f"an audio_values Tensor, got {type(audio_values).__name__}"
            )
        if return_dict:
            return GenerationOutput(audio_values=audio_values, raw_output=raw_output)
        return audio_values



# Backward compatibility for the spelling used in the initial project task.
MusciGen = MusicGen

__all__ = [
    "GenerationOutput",
    "HiddenStateHookContext",
    "MusicGen",
    "MusicGenOutput",
    "MusciGen",
]
