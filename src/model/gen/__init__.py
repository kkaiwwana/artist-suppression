"""Pure PyTorch MusicGen wrapper."""

from .musicgen import (
    GenerationOutput,
    HiddenStateHookContext,
    MusicGen,
    MusicGenOutput,
)

__all__ = [
    "GenerationOutput",
    "HiddenStateHookContext",
    "MusicGen",
    "MusicGenOutput"
]
