"""
Temporal Transformer for gesture recognition.

Treats each time step as a token (pose + optflow features), runs TransformerEncoder,
then pools over time (masked mean) and projects to class logits.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    in_features: int = 78
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 2048


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence length up to max_len."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) — use only first T positions
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer over time for gesture sequences.

    Input: batch dict with "x" (B, T, F) and "lengths" (B,), or legacy "pose" (B,T,72), "optflow" (B,T,6), "length" (B,).
    Output: logits (B, num_classes).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        num_classes = cfg.num_classes
        in_features = cfg.in_features
        d_model = cfg.d_model
        nhead = cfg.nhead
        num_encoder_layers = cfg.num_encoder_layers
        dim_feedforward = cfg.dim_feedforward
        dropout = cfg.dropout
        max_seq_len = cfg.max_seq_len

        self.num_classes = num_classes
        self.d_model = d_model
        self.input_dim = in_features  # alias for backward compat

        self.input_norm_raw = nn.LayerNorm(in_features)
        self.input_proj = nn.Linear(in_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pool_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Classifier head: smaller init for stability
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.5)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        batch: Optional[dict[str, Any]] = None,
        *,
        x: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        optflow: Optional[torch.Tensor] = None,
        length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass. Prefers unified batch with "x" and "lengths"; supports legacy "pose", "optflow", "length".
        """
        if batch is not None:
            if "x" in batch and "lengths" in batch:
                x = batch["x"]
                lengths = batch["lengths"]
            elif "pose" in batch and "optflow" in batch:
                pose = batch["pose"]
                optflow = batch["optflow"]
                length = batch.get("length", batch.get("lengths"))

        if x is not None and lengths is not None:
            # Unified format: x (B, T, F)
            pass
        elif pose is not None and optflow is not None and length is not None:
            x = torch.cat([pose, optflow], dim=-1)
            lengths = length
        else:
            raise ValueError("Provide batch with 'x' and 'lengths', or 'pose', 'optflow', 'length'")

        B, T, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.input_norm_raw(x)
        x = self.input_norm(self.input_proj(x))
        x = self.pos_enc(x)

        key_padding = length_to_padding_mask(lengths, T, device=x.device)

        x = self.transformer(x, src_key_padding_mask=key_padding)  # (B, T, d_model)
        x = self.dropout(x)

        # Masked mean over time
        key_padding_float = key_padding.float().unsqueeze(-1)  # (B, T, 1)
        x_masked = x.masked_fill(key_padding_float.bool(), 0.0)
        lengths_clamped = lengths.clamp(min=1).unsqueeze(1).float()  # (B, 1)
        pooled = x_masked.sum(dim=1) / lengths_clamped  # (B, d_model)
        pooled = self.pool_norm(pooled)

        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


def length_to_padding_mask(length: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    """Create key_padding_mask (B, T) where True = padded position."""
    B = length.size(0)
    arange = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
    return arange >= length.unsqueeze(1)  # (B, T)
