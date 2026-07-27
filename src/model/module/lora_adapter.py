"""Standard linear LoRA adapters for MusicGen attention projections.

The adapter follows the original LoRA parameterisation for a frozen linear
projection ``W``::

    y = W x + (alpha / rank) * B(A(dropout(x)))

``A`` is Kaiming-initialised and ``B`` is zero-initialised by default, so
injecting LoRA is an exact identity transformation at initialisation.  There
is deliberately no activation between ``A`` and ``B``.

MusicGen defaults to adapting the query and value projections in every
decoder self-attention and cross-attention module.  Query/value is the common
LoRA choice: query updates change which context is selected, while value
updates change the content written back to the residual stream.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence

from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Wrap a frozen :class:`~torch.nn.Linear` with a trainable LoRA branch.

    Args:
        base_layer: Pretrained projection retained as the frozen base path.
        rank: Low-rank dimension.
        alpha: Scaling numerator. The effective multiplier is ``alpha/rank``.
        dropout: Dropout applied only to the LoRA branch input.
        zero_init: Zero-initialise ``B`` so the initial forward is identical
            to the base layer. This should normally remain true.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int = 8,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be torch.nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.zero_init = bool(zero_init)

        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        self.reset_lora_parameters()

        reference = base_layer.weight
        self.lora_A.to(device=reference.device, dtype=reference.dtype)
        self.lora_B.to(device=reference.device, dtype=reference.dtype)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def reset_lora_parameters(self) -> None:
        """Initialise ``A`` and ``B`` using the standard LoRA scheme."""

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        if self.zero_init:
            nn.init.zeros_(self.lora_B.weight)
        else:
            nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

    def lora_residual(self, x: Tensor) -> Tensor:
        """Return only the scaled low-rank update for diagnostics/tests."""

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.lora_residual(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}, "
            f"zero_init={self.zero_init}"
        )


def _is_attention_module(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return "attention" in class_name


def _layer_index(qualified_name: str) -> Optional[int]:
    """Extract a transformer layer index from a qualified module name."""

    match = re.search(
        r"(?:^|\.)(?:layers|blocks|decoder_layers)\.(\d+)(?:\.|$)",
        qualified_name,
    )
    return int(match.group(1)) if match else None


def inject_attention_lora(
    root: nn.Module,
    *,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    zero_init: bool = True,
    targets: Sequence[str] = ("q_proj", "v_proj"),
    layers: Optional[Sequence[int]] = None,
    attention_types: Optional[Sequence[str]] = None,
    include: Optional[Iterable[str]] = None,
) -> List[str]:
    """Inject LoRA into selected projections of decoder attention modules.

    ``targets`` contains attention child names such as ``q_proj``, ``k_proj``,
    ``v_proj`` or ``out_proj``. ``layers`` optionally selects decoder layer
    indices. ``attention_types`` optionally selects attention modules by the
    final component of their qualified name, for example ``self_attn`` or
    ``encoder_attn``. ``include`` optionally restricts attention modules by
    their full qualified names under ``root``. Calling this function twice is
    idempotent.

    Returns:
        Qualified projection names that are backed by :class:`LoRALinear`.
    """

    target_names = tuple(str(name).strip() for name in targets)
    if not target_names or any(not name for name in target_names):
        raise ValueError("targets must contain at least one non-empty name")
    if len(set(target_names)) != len(target_names):
        raise ValueError("targets must not contain duplicates")

    selected_layers: Optional[tuple[int, ...]] = None
    if layers is not None:
        if not layers:
            raise ValueError("layers must be null or contain at least one index")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in layers):
            raise TypeError("layers must contain Python integers")
        selected_layers = tuple(int(index) for index in layers)
        if any(index < 0 for index in selected_layers):
            raise ValueError("layers must contain non-negative indices")
        if len(set(selected_layers)) != len(selected_layers):
            raise ValueError("layers must not contain duplicates")

    selected_attention_types: Optional[tuple[str, ...]] = None
    if attention_types is not None:
        selected_attention_types = tuple(
            str(attention_type).strip() for attention_type in attention_types
        )
        if not selected_attention_types or any(
            not attention_type for attention_type in selected_attention_types
        ):
            raise ValueError(
                "attention_types must be null or contain at least one non-empty name"
            )
        if len(set(selected_attention_types)) != len(selected_attention_types):
            raise ValueError("attention_types must not contain duplicates")

    allowed = set(include) if include is not None else None
    attention_modules = [
        (name, module)
        for name, module in root.named_modules()
        if name
        and (allowed is None or name in allowed)
        and _is_attention_module(module)
    ]
    if selected_attention_types is not None:
        available_attention_types = {
            name.rsplit(".", 1)[-1] for name, _ in attention_modules
        }
        missing_attention_types = sorted(
            set(selected_attention_types) - available_attention_types
        )
        if missing_attention_types:
            raise ValueError(
                "requested attention types do not exist: "
                f"{missing_attention_types}; "
                f"available={sorted(available_attention_types)}"
            )
        selected = set(selected_attention_types)
        attention_modules = [
            (name, module)
            for name, module in attention_modules
            if name.rsplit(".", 1)[-1] in selected
        ]
    if selected_layers is not None:
        available_layers = {
            index
            for name, _ in attention_modules
            if (index := _layer_index(name)) is not None
        }
        missing_layers = sorted(set(selected_layers) - available_layers)
        if missing_layers:
            raise ValueError(
                f"requested LoRA decoder layers do not exist: {missing_layers}; "
                f"available={sorted(available_layers)}"
            )
        selected = set(selected_layers)
        attention_modules = [
            (name, module)
            for name, module in attention_modules
            if _layer_index(name) in selected
        ]

    for _, attention in attention_modules:
        for target_name in target_names:
            projection = getattr(attention, target_name, None)
            if isinstance(projection, LoRALinear):
                continue
            if not isinstance(projection, nn.Linear):
                continue
            setattr(
                attention,
                target_name,
                LoRALinear(
                    projection,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    zero_init=zero_init,
                ),
            )

    return [
        name
        for name, module in root.named_modules()
        if name and isinstance(module, LoRALinear)
    ]


def freeze_module_except_lora(module: nn.Module) -> None:
    """Freeze ``module`` and leave only LoRA ``A``/``B`` trainable."""

    for parameter in module.parameters():
        parameter.requires_grad = False
    for submodule in module.modules():
        if isinstance(submodule, LoRALinear):
            for parameter in submodule.lora_A.parameters():
                parameter.requires_grad = True
            for parameter in submodule.lora_B.parameters():
                parameter.requires_grad = True


# Keep the historical helper names as API aliases while changing their
# semantics to authentic linear LoRA. New code should use the explicit names.
inject_attention_adapters = inject_attention_lora
freeze_module_except_adapters = freeze_module_except_lora


__all__ = [
    "LoRALinear",
    "freeze_module_except_adapters",
    "freeze_module_except_lora",
    "inject_attention_adapters",
    "inject_attention_lora",
]
