"""
Stage 3 — Multi-Sensor Temporal Encoders for the CBDL Pipeline.

Each encoder takes a sensor stream of shape (B, C, T) and returns a
128-dimensional embedding vector (B, 128) using a CNN + BiLSTM backbone.

Encoders:
  - FingerTemporalEncoder: finger IMU (6 channels)
  - GaitTemporalEncoder: gait IMU (6 channels)
  - BalanceEncoder: insole_pressure + phone_acc_gyro (10 channels)
  - SensorEncoder: wrist + phone streams with attention pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """
    1D convolutional block: Conv1d → BatchNorm → ReLU → Dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C_in, T). Returns: (B, C_out, T)."""
        return self.dropout(F.relu(self.bn(self.conv(x))))


class CNNBiLSTMEncoder(nn.Module):
    """
    Shared backbone: 3-block CNN + 2-layer BiLSTM + mean-pool + Linear.

    Args:
        in_channels: Number of input sensor channels.
        embed_dim: Output embedding dimension (default 128).
        lstm_hidden: BiLSTM hidden size.
        lstm_layers: Number of BiLSTM layers.
        dropout: Dropout rate.
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=5, dropout=dropout),
            ConvBlock(32, 64, kernel_size=5, dropout=dropout),
            ConvBlock(64, 128, kernel_size=5, dropout=dropout),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(lstm_hidden * 2, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, T). Returns: (B, embed_dim)."""
        feat = self.conv_blocks(x)
        feat = feat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        pooled = lstm_out.mean(dim=1)
        return self.proj(pooled)


class FingerTemporalEncoder(nn.Module):
    """Encoder for finger IMU (6 channels). Returns (B, embed_dim)."""

    def __init__(self, embed_dim: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = CNNBiLSTMEncoder(6, embed_dim, lstm_hidden, lstm_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 6, T). Returns: (B, embed_dim)."""
        return self.backbone(x)


class GaitTemporalEncoder(nn.Module):
    """Encoder for gait shoe-IMU (6 channels). Returns (B, embed_dim)."""

    def __init__(self, embed_dim: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = CNNBiLSTMEncoder(6, embed_dim, lstm_hidden, lstm_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 6, T). Returns: (B, embed_dim)."""
        return self.backbone(x)


class BalanceEncoder(nn.Module):
    """Encoder for insole pressure (4 ch) + phone acc/gyro (6 ch). Returns (B, embed_dim)."""

    def __init__(self, embed_dim: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = CNNBiLSTMEncoder(10, embed_dim, lstm_hidden, lstm_layers, dropout)

    def forward(self, insole: torch.Tensor, phone: torch.Tensor) -> torch.Tensor:
        """Args: insole (B, 4, T), phone (B, 6, T). Returns: (B, embed_dim)."""
        x = torch.cat([insole, phone], dim=1)
        return self.backbone(x)


class SensorEncoder(nn.Module):
    """Encoder for wrist + phone streams with cross-attention pooling. Returns (B, embed_dim)."""

    def __init__(self, embed_dim: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2, n_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.wrist_enc = CNNBiLSTMEncoder(6, embed_dim, lstm_hidden, lstm_layers, dropout)
        self.phone_enc = CNNBiLSTMEncoder(6, embed_dim, lstm_hidden, lstm_layers, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, wrist: torch.Tensor, phone: torch.Tensor) -> torch.Tensor:
        """Args: wrist (B, 6, T), phone (B, 6, T). Returns: (B, embed_dim)."""
        w_emb = self.wrist_enc(wrist).unsqueeze(1)
        p_emb = self.phone_enc(phone).unsqueeze(1)
        tokens = torch.cat([w_emb, p_emb], dim=1)
        attended, _ = self.attn(tokens, tokens, tokens)
        return self.proj(attended.mean(dim=1))


class MultiStreamEncoder(nn.Module):
    """
    Combines all four stream encoders.

    Returns four embeddings: finger, gait, balance, sensor.

    Args:
        embed_dim: Per-stream embedding dimension (default 128).
        lstm_hidden: BiLSTM hidden size.
        lstm_layers: Number of BiLSTM layers.
        n_heads: Attention heads for SensorEncoder.
        dropout: Dropout rate.
    """

    def __init__(self, embed_dim: int = 128, lstm_hidden: int = 128, lstm_layers: int = 2, n_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.finger_enc = FingerTemporalEncoder(embed_dim, lstm_hidden, lstm_layers, dropout)
        self.gait_enc = GaitTemporalEncoder(embed_dim, lstm_hidden, lstm_layers, dropout)
        self.balance_enc = BalanceEncoder(embed_dim, lstm_hidden, lstm_layers, dropout)
        self.sensor_enc = SensorEncoder(embed_dim, lstm_hidden, lstm_layers, n_heads, dropout)
        self.embed_dim = embed_dim

    def forward(self, finger: torch.Tensor, gait: torch.Tensor, insole: torch.Tensor, phone: torch.Tensor, wrist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode all streams.

        Args:
            finger (B, 6, T), gait (B, 6, T), insole (B, 4, T),
            phone (B, 6, T), wrist (B, 6, T).

        Returns:
            Tuple of (finger_emb, gait_emb, balance_emb, sensor_emb), each (B, 128).
        """
        return (
            self.finger_enc(finger),
            self.gait_enc(gait),
            self.balance_enc(insole, phone),
            self.sensor_enc(wrist, phone),
        )
