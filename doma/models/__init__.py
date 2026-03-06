"""
Gesture recognition models.

- TemporalTransformer: transformer over time steps (pose + optflow sequence) -> class logits.
"""

from doma.models.temporal_transformer import TemporalTransformer

__all__ = ["TemporalTransformer"]
