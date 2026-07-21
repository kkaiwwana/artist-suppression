"""Training callbacks."""

from src.callback.audio_comparison import ValidationAudioComparisonCallback
from src.callback.git_diff import GitDiffCallback

__all__ = ["GitDiffCallback", "ValidationAudioComparisonCallback"]
