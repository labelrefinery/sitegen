"""Synthetic construction-site MCAP generation with held-out ground truth."""

from .scene import Scene
from .writer import generate

__all__ = ["Scene", "generate"]
