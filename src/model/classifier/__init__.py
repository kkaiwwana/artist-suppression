"""Transferred music-audio artist classifiers."""

from .lightning import ArtistClassifierLightningModule
from .mert import MERTArtistClassifier

__all__ = ["ArtistClassifierLightningModule", "MERTArtistClassifier"]
