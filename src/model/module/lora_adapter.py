"""Non-linear LoRA adapters for attention outputs.

The adapter in this file is deliberately independent of Transformers and
PyTorch Lightning.  It is a residual bottleneck branch::

    x -> dropout -> down -> activation -> up -> scaling

The ``up`` projection is zero-initialised by default.  Consequently an
adapter-enabled model is functionally identical to the base model at
initialisation, while the adapter parameters remain trainable.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn


def _activation(name: str) -> nn.Module:
    """Build a small activation module from a user-facing name."""

    normalized = name.lower().replace("-", "_")
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(
        "activation must be one of gelu, silu, relu, tanh, identity; "
        f"got {name!r}"
    )


class NonlinearLoRA(nn.Module):
    """A zero-initialised non-linear low-rank residual adapter.

    Args:
        in_features: Last-dimension size of the attention output.
        out_features: Last-dimension size of the residual.  MusicGen
            attention adapters normally use the same value as ``in_features``.
        rank: Bottleneck rank.
        alpha: LoRA scaling numerator.  The effective scale is ``alpha/rank``.
        activation: Non-linearity between the down and up projections.
        dropout: Dropout applied before the down projection.
        zero_init: Zero-initialise the up projection when true.

    The module accepts tensors of shape ``[..., in_features]`` and returns a
    tensor of shape ``[..., out_features]``.  It returns the residual branch,
    not ``x + residual``; :class:`AttentionOutputAdapter` performs the
    residual addition around an existing attention module.
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        rank: int = 8,
        alpha: Optional[float] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features is None:
            out_features = in_features
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.zero_init = bool(zero_init)

        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(self.in_features, self.rank, bias=False)
        self.activation = _activation(activation)
        self.up = nn.Linear(self.rank, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        if self.zero_init:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.kaiming_uniform_(self.up.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        return self.up(self.activation(self.down(self.dropout(x)))) * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}, "
            f"zero_init={self.zero_init}"
        )


# The longer name is useful in configuration files and keeps compatibility
# with callers that prefer to call this an adapter rather than LoRA.
NonlinearLoRAAdapter = NonlinearLoRA


class AttentionOutputAdapter(nn.Module):
    """Wrap an attention module and add a trainable residual after it.

    Transformers attention modules generally return ``(hidden_states, ...)``
    when attention weights or cache values are involved.  This wrapper updates
    only the first element and preserves the remaining return values.
    """

    def __init__(self, attention: nn.Module, adapter: NonlinearLoRA) -> None:
        super().__init__()
        self.attention = attention
        self.adapter = adapter

    @staticmethod
    def _replace_first(output: object, residual_fn) -> object:
        if isinstance(output, Tensor):
            return output + residual_fn(output)
        if isinstance(output, tuple):
            if not output or not isinstance(output[0], Tensor):
                return output
            return (output[0] + residual_fn(output[0]),) + output[1:]
        if isinstance(output, list):
            if not output or not isinstance(output[0], Tensor):
                return output
            return [output[0] + residual_fn(output[0]), *output[1:]]
        # This is uncommon for attention modules, but gives a useful error
        # instead of silently disabling the adapter for a custom module.
        raise TypeError(
            "attention output must be a Tensor, tuple, or list whose first "
            f"item is a Tensor; got {type(output).__name__}"
        )

    def forward(self, *args, **kwargs):
        output = self.attention(*args, **kwargs)
        return self._replace_first(output, self.adapter)


def _is_attention_module(module: nn.Module) -> bool:
    if isinstance(module, nn.MultiheadAttention):
        return True
    class_name = module.__class__.__name__.lower()
    return "attention" in class_name and not isinstance(module, AttentionOutputAdapter)


def _attention_dim(module: nn.Module) -> int:
    for attribute in ("embed_dim", "hidden_size", "d_model"):
        value = getattr(module, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    for attribute in ("q_proj", "out_proj", "q_proj"):
        projection = getattr(module, attribute, None)
        value = getattr(projection, "in_features", None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        f"cannot infer hidden size for attention module {module.__class__.__name__}"
    )


def _get_parent(root: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_attention_adapters(
    root: nn.Module,
    *,
    rank: int = 8,
    alpha: Optional[float] = None,
    activation: str = "gelu",
    dropout: float = 0.0,
    zero_init: bool = True,
    include: Optional[Iterable[str]] = None,
) -> List[str]:
    """Insert adapters after every matching attention module in ``root``.

    ``include`` optionally contains qualified module names.  When omitted, all
    attention modules under ``root`` are wrapped.  The MusicGen wrapper passes
    the decoder as ``root`` so the frozen T5 text encoder is not modified.

    Returns the qualified names of the wrapped attention modules.  Calling the
    function twice on the same root is safe and does not double-wrap modules.
    """

    allowed = set(include) if include is not None else None
    names = [
        name
        for name, module in root.named_modules()
        if name
        and (allowed is None or name in allowed)
        and _is_attention_module(module)
    ]

    for name in names:
        parent, child_name = _get_parent(root, name)
        attention = getattr(parent, child_name)
        if isinstance(attention, AttentionOutputAdapter):
            continue
        dim = _attention_dim(attention)
        adapter = NonlinearLoRA(
            in_features=dim,
            out_features=dim,
            rank=rank,
            alpha=alpha,
            activation=activation,
            dropout=dropout,
            zero_init=zero_init,
        )
        reference_parameter = next(attention.parameters(), None)
        if reference_parameter is not None:
            # Adapters are created after a pretrained model may already have
            # been loaded in fp16/bf16 or placed on an accelerator.  Match the
            # wrapped attention immediately so its first forward cannot mix
            # devices or floating-point dtypes.
            adapter = adapter.to(
                device=reference_parameter.device,
                dtype=reference_parameter.dtype,
            )
        setattr(parent, child_name, AttentionOutputAdapter(attention, adapter))

    return [
        name
        for name, module in root.named_modules()
        if isinstance(module, AttentionOutputAdapter)
    ]


def freeze_module_except_adapters(module: nn.Module) -> None:
    """Freeze ``module`` while keeping all :class:`NonlinearLoRA` trainable."""

    for parameter in module.parameters():
        parameter.requires_grad = False
    for submodule in module.modules():
        if isinstance(submodule, NonlinearLoRA):
            for parameter in submodule.parameters():
                parameter.requires_grad = True


__all__ = [
    "AttentionOutputAdapter",
    "NonlinearLoRA",
    "NonlinearLoRAAdapter",
    "freeze_module_except_adapters",
    "inject_attention_adapters",
]
