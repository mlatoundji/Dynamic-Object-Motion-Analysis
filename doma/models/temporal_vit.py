"""
Vision Transformer (ViT) for gesture recognition from temporal flow clips.
Encodes each frame with a shared ViT backbone, then aggregates over time with a small transformer.
Accepts batch dict from DataLoader: "frames" (B, T, C, H, W), "label", "length".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import timm


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    num_frames: int = 8
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    temporal_depth: int = 2
    drop_rate: float = 0.0
    pretrained: bool = True


class TemporalViT(nn.Module):
    """
    ViT-based gesture classifier for flow clips.
    Input: batch dict with "frames" (B, T, C, H, W), or tensor x (B, T, C, H, W).
    Output: (B, num_classes) logits.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        num_classes = cfg.num_classes
        num_frames = cfg.num_frames
        img_size = cfg.img_size
        embed_dim = cfg.embed_dim
        num_heads = cfg.num_heads
        temporal_depth = cfg.temporal_depth
        drop_rate = cfg.drop_rate
        pretrained = cfg.pretrained

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Per-frame encoder: ViT backbone (no head)
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=img_size,
        )
        # Ensure we get embed_dim (vit_small has 384)
        self.frame_embed_dim = self.backbone.embed_dim

        # Temporal transformer
        # Learnable temporal positional embedding
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, self.frame_embed_dim))
        nn.init.normal_(self.temporal_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.frame_embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=temporal_depth)
        self.temporal_norm = nn.LayerNorm(self.frame_embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(self.frame_embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.frame_embed_dim, num_classes),
        )

    def forward(
        self,
        batch: Optional[dict[str, Any]] = None,
        *,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x or batch["frames"]: (B, T, C, H, W)
        """
        if batch is not None:
            x = batch["frames"]
        if x is None:
            raise ValueError("Provide either batch dict or x (frames tensor)")
        B, T, C, H, W = x.shape
        # Encode each frame
        x = x.view(B * T, C, H, W)
        frame_tokens = self.backbone(x)  # (B*T, num_patches, embed_dim)
        # Use CLS or mean over patches; vit_small_patch16_224 global_pool="" returns all patch tokens
        frame_repr = frame_tokens.mean(dim=1)  # (B*T, embed_dim)
        # Restore original batch and time dimensions
        frame_repr = frame_repr.view(B, T, self.frame_embed_dim)

        # Temporal transformer
        frame_repr = frame_repr + self.temporal_embed[:, :T] # (B, T, embed_dim)
        out = self.temporal_transformer(frame_repr)  # (B, T, embed_dim)
        out = self.temporal_norm(out.mean(dim=1))  # (B, embed_dim)

        return self.head(out)
