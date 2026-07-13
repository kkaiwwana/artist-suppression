"""Compatibility namespace for model modules."""

from .lora_adapter import (
    AttentionOutputAdapter,
    NonlinearLoRA,
    NonlinearLoRAAdapter,
    freeze_module_except_adapters,
    inject_attention_adapters,
)

__all__ = [
    "AttentionOutputAdapter",
    "NonlinearLoRA",
    "NonlinearLoRAAdapter",
    "freeze_module_except_adapters",
    "inject_attention_adapters",
]
