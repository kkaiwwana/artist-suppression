"""Compatibility namespace for model modules."""

from .lora_adapter import (
    freeze_module_except_adapters,
    freeze_module_except_lora,
    inject_attention_adapters,
    inject_attention_lora,
    LoRALinear,
)

__all__ = [
    "freeze_module_except_adapters",
    "freeze_module_except_lora",
    "inject_attention_adapters",
    "inject_attention_lora",
    "LoRALinear",
]
