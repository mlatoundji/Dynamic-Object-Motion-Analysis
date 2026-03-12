"""
CNN-LSTM temporal classifier for gesture recognition (manifest-based features).

Conv1D over time + LSTM + masked mean pooling. Training via doma-train (doma.modeling.train).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    in_features: int
    conv_channels: int = 128
    conv_layers: int = 2
    conv_kernel: int = 5
    conv_dropout: float = 0.1
    lstm_hidden: int = 256
    lstm_layers: int = 1
    bidirectional: bool = True
    lstm_dropout: float = 0.1
    head_dropout: float = 0.2


def _masked_mean(x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
    """x: (B,T,C), lengths: (B,) -> (B,C)."""
    B, T, C = x.shape
    if T == 0:
        return x.new_zeros((B, C))
    device = x.device
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    mask = idx < lengths.clamp_min(0).unsqueeze(1)
    mask_f = mask.to(dtype=x.dtype).unsqueeze(2)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


class ConvBlock(nn.Module):
    def __init__(self, *, in_ch: int, out_ch: int, kernel: int, dropout: float) -> None:
        super().__init__()
        pad = int((kernel - 1) // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=int(kernel), padding=int(pad)),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class CNNLSTM(nn.Module):
    """
    Temporal CNN (Conv1D over time) + LSTM classifier.
    Input: x (B,T,F), lengths (B,), or batch=dict with "x", "lengths". Output: logits (B,num_classes).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to use CNNLSTM.")
        super().__init__()
        self.cfg = cfg

        conv_layers = int(max(0, cfg.conv_layers))
        conv_ch = int(cfg.conv_channels)
        kernel = int(cfg.conv_kernel)
        drop = float(cfg.conv_dropout)

        blocks = []
        in_ch = int(cfg.in_features)
        for _ in range(conv_layers):
            blocks.append(ConvBlock(in_ch=in_ch, out_ch=conv_ch, kernel=kernel, dropout=drop))
            in_ch = conv_ch
        self.conv = nn.Sequential(*blocks) if blocks else nn.Identity()

        lstm_in = in_ch
        self.lstm = nn.LSTM(
            input_size=int(lstm_in),
            hidden_size=int(cfg.lstm_hidden),
            num_layers=int(cfg.lstm_layers),
            dropout=float(cfg.lstm_dropout) if int(cfg.lstm_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=bool(cfg.bidirectional),
        )
        lstm_out = int(cfg.lstm_hidden) * (2 if cfg.bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Dropout(float(cfg.head_dropout)),
            nn.Linear(lstm_out, int(cfg.num_classes)),
        )

    def forward(
        self,
        x: "torch.Tensor | None" = None,
        lengths: "torch.Tensor | None" = None,
        *,
        batch: "dict[str, torch.Tensor] | None" = None,
    ) -> "torch.Tensor":
        if batch is not None:
            x = batch["x"]
            lengths = batch["lengths"]
        if x is None or lengths is None:
            raise ValueError("CNNLSTM requires (x, lengths) or batch= with 'x' and 'lengths'")
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        lengths_cpu = lengths.to(dtype=torch.long, device="cpu").clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        pooled = _masked_mean(out, lengths.to(device=out.device))
        return self.head(pooled)


def export_onnx(
    model: "CNNLSTM",
    *,
    out_path,
    opset: int = 17,
    max_len: int = 256,
) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export.")
    out_path = str(out_path)
    model.eval()
    B, F = 1, int(model.cfg.in_features)
    dummy_x = torch.zeros((B, int(max_len), F), dtype=torch.float32)
    dummy_len = torch.tensor([int(max_len)], dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_x, dummy_len),
        out_path,
        input_names=["x", "lengths"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "batch", 1: "time"},
            "lengths": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=int(opset),
    )
