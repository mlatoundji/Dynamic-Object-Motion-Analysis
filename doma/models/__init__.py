"""
Gesture recognition models.

- TemporalTransformer: transformer over time steps (pose + optflow sequence) -> class logits.
- TemporalViT: patch-embed flow/RGB frames + temporal transformer -> class logits.
- CNNLSTM: Conv1D + LSTM (manifest-based features); accepts (x, lengths) or batch=.
"""

from doma.models.temporal_transformer import TemporalTransformer
from doma.models.temporal_vit import TemporalViT
from doma.models.cnn_lstm import CNNLSTM, ModelConfig

__all__ = ["TemporalTransformer", "TemporalViT", "CNNLSTM", "ModelConfig"]
