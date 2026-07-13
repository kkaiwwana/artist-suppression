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
    ``[batch * num_codebooks, audio_length]``.
``labels`` (optional)
    Target EnCodec codes in the same ``[batch, num_codebooks, audio_length]``
    layout.  If omitted by ``training_step``, ``audio_tokens`` are used as
    labels and the original MusicGen cross-entropy is computed by the wrapped
    model.
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
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.module.lora_adapter import (
    freeze_module_except_adapters,
    inject_attention_adapters,
)


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
    use :meth:`from_pretrained` for the normal text-to-music workflow.  When
    ``use_adapter=True``, every attention module in the decoder is wrapped by a
    zero-initialised non-linear LoRA branch.  Unless explicitly overridden,
    this also freezes the original model parameters.

    Args:
        backbone: A ``MusicgenForConditionalGeneration`` instance.  Passing a
            model object keeps the class usable with locally constructed tiny
            configs in tests and notebooks.
        processor: An ``AutoProcessor`` instance used by ``generate``.
        use_adapter: Insert a non-linear LoRA after every decoder attention.
        freeze_backbone: Freeze the base model.  Defaults to ``use_adapter``.
        adapter_rank: Bottleneck rank for each adapter.
        adapter_alpha: LoRA alpha.  Defaults to the rank.
        adapter_activation: Activation in the adapter bottleneck.
        adapter_dropout: Dropout before the adapter down projection.
        adapter_zero_init: Zero-initialise the adapter up projection.
    """

    def __init__(
        self,
        backbone: nn.Module,
        processor: Optional[Any] = None,
        *,
        use_adapter: bool = False,
        freeze_backbone: Optional[bool] = None,
        adapter_rank: int = 8,
        adapter_alpha: Optional[float] = None,
        adapter_activation: str = "gelu",
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")

        self.backbone = backbone
        self.processor = processor
        self.use_adapter = bool(use_adapter)
        self.adapter_module_names: list[str] = []
        self._hidden_state_handles: List[Any] = []

        if self.use_adapter:
            adapter_root = self._find_decoder_root(backbone)
            self.adapter_module_names = inject_attention_adapters(
                adapter_root,
                rank=adapter_rank,
                alpha=adapter_alpha,
                activation=adapter_activation,
                dropout=adapter_dropout,
                zero_init=adapter_zero_init,
            )
            if not self.adapter_module_names:
                raise RuntimeError(
                    "use_adapter=True but no decoder attention modules were found"
                )

        if freeze_backbone is None:
            freeze_backbone = self.use_adapter
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            freeze_module_except_adapters(self.backbone)

        self.num_codebooks = self._read_num_codebooks(backbone)
        self.pad_token_id = self._read_config_value("pad_token_id")
        self.audio_sample_rate = self._read_config_value(
            "sampling_rate", config_path=("audio_encoder",)
        ) or 32000

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
        adapter_activation: str = "gelu",
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
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
            adapter_activation=adapter_activation,
            adapter_dropout=adapter_dropout,
            adapter_zero_init=adapter_zero_init,
        )
        if device is not None:
            wrapped.to(device)
        return wrapped

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

    @property
    def trainable_parameters(self):
        """Iterator over parameters visible to an outer optimizer."""

        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters)

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
            for handle in handles:
                handle.remove()
                if handle in self._hidden_state_handles:
                    self._hidden_state_handles.remove(handle)

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
            for handle in handles:
                handle.remove()
                if handle in self._hidden_state_handles:
                    self._hidden_state_handles.remove(handle)

    def register_concept_learner_hooks(
        self,
        concept_learner: nn.Module,
        condition: Optional[Any] = None,
        module_names: Optional[str | int | Sequence[str | int]] = None,
        *,
        condition_kwargs: Optional[Dict[str, Any]] = None,
        intervention_scale: float | Tensor = 1.0,
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

        def _apply(hidden_state: Tensor, context: HiddenStateHookContext) -> Tensor:
            return concept_learner(
                hidden_state,
                block_index=context.block_index,
                condition=condition,
                intervention_scale=intervention_scale,
            )

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
        decoder_input_ids = self._first(values, "decoder_input_ids", "audio_tokens")
        labels = self._first(values, "labels", "audio_labels", "target_audio_tokens")
        if labels is None and use_audio_as_labels:
            labels = decoder_input_ids

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
        if standard_labels is not None:
            payload["labels"] = standard_labels.transpose(1, 2).contiguous()

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
        return {
            "loss": output.loss,
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
        return {
            "loss": output.loss,
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



__all__ = ["GenerationOutput", "MusicGen", "MusicGenOutput", "HiddenStateHookContext"]
