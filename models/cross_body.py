"""
Stage 4 — Cross-Body Dependency Learning (CBDL Core).

Five dependency modules:
  1. CrossLimbAttention — MHA over 4 tokens
  2. TemporalDependencyGraph — custom pure-PyTorch 2-layer GAT
  3. DynamicDependencyMatrix — GRU-based 4×4 coupling matrix
  4. TimeLagCorrelation — learnable lag estimation
  5. ContrastiveDependencyLoss — InfoNCE loss

CrossBodyDependencyModule produces a 512-d fused embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossLimbAttention(nn.Module):
    """Multi-head self-attention over the 4 stream tokens. Returns (attended, weights)."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, 4, embed_dim).
        Returns:
            attended (B, 4, embed_dim), weights (B, 4, 4).
        """
        attended, weights = self.attn(tokens, tokens, tokens, average_attn_weights=True)
        return self.norm(attended + tokens), weights


class GATLayer(nn.Module):
    """Single GAT layer (pure PyTorch, no PyG). Operates on fully-connected graph of 4 nodes."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, N, in_dim). Returns: (B, N, out_dim)."""
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.n_heads, self.head_dim)
        h_i = h.unsqueeze(2).expand(B, N, N, self.n_heads, self.head_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, self.n_heads, self.head_dim)
        pair = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky((pair * self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1))
        alpha = self.dropout(F.softmax(e, dim=2))
        out = (alpha.unsqueeze(-1) * h_j).sum(dim=2).view(B, N, self.out_dim)
        return F.elu(out)


class TemporalDependencyGraph(nn.Module):
    """Two-layer GAT over 4 body-part nodes."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.layer1 = GATLayer(embed_dim, embed_dim, n_heads, dropout)
        self.layer2 = GATLayer(embed_dim, embed_dim, n_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: tokens (B, 4, embed_dim). Returns: (B, 4, embed_dim)."""
        return self.norm(self.layer2(self.layer1(tokens)) + tokens)


class DynamicDependencyMatrix(nn.Module):
    """GRU-based head producing a 4×4 symmetric coupling matrix."""

    def __init__(self, embed_dim: int = 128, hidden_size: int = 256) -> None:
        super().__init__()
        self.gru = nn.GRU(embed_dim * 4, hidden_size, num_layers=2, batch_first=True)
        self.proj = nn.Linear(hidden_size, 16)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: tokens (B, 4, embed_dim). Returns: matrix (B, 4, 4) symmetric, [0,1]."""
        B = tokens.shape[0]
        _, h = self.gru(tokens.reshape(B, 1, -1))
        matrix = self.proj(h[-1]).view(B, 4, 4)
        matrix = (matrix + matrix.transpose(-1, -2)) * 0.5
        return torch.sigmoid(matrix)


class TimeLagCorrelation(nn.Module):
    """Learnable lag estimation between finger and gait (0–max_lag samples)."""

    def __init__(self, embed_dim: int = 128, max_lag: int = 25) -> None:
        super().__init__()
        self.max_lag = max_lag
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.lag_emb = nn.Embedding(max_lag + 1, embed_dim)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, finger_emb: torch.Tensor, gait_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            finger_emb (B, D), gait_emb (B, D).
        Returns:
            fused (B, D), lag_weights (B, max_lag+1).
        """
        query = self.query_proj(finger_emb)
        lags = torch.arange(self.max_lag + 1, device=finger_emb.device)
        lag_embs = self.lag_emb(lags)
        scores = torch.matmul(query, lag_embs.T)
        lag_weights = F.softmax(scores / (query.shape[-1] ** 0.5), dim=-1)
        lag_context = torch.matmul(lag_weights, lag_embs)
        fused = self.out_proj(torch.cat([gait_emb, lag_context], dim=-1))
        return fused, lag_weights


class ContrastiveDependencyLoss(nn.Module):
    """InfoNCE loss for positive/negative stream embedding pairs."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """Args: anchor (B, D), positive (B, D). Returns: scalar InfoNCE loss."""
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        logits = torch.matmul(a, p.T) / self.temperature
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return F.cross_entropy(logits, labels)


class CrossBodyDependencyModule(nn.Module):
    """
    Full Stage 4 module.

    Stacks all five dependency modules and produces a 512-d cross-body embedding.

    Args:
        embed_dim: Per-stream embedding dimension (default 128).
        n_heads: Number of attention heads.
        max_lag: Maximum time lag in samples.
        dropout: Dropout rate.
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, max_lag: int = 25, dropout: float = 0.2) -> None:
        super().__init__()
        self.cross_limb_attn = CrossLimbAttention(embed_dim, n_heads, dropout)
        self.tdg = TemporalDependencyGraph(embed_dim, n_heads, dropout)
        self.ddm = DynamicDependencyMatrix(embed_dim)
        self.time_lag = TimeLagCorrelation(embed_dim, max_lag)
        self.contrastive_loss_fn = ContrastiveDependencyLoss(temperature=0.07)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, finger: torch.Tensor, gait: torch.Tensor, balance: torch.Tensor, sensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run all cross-body dependency modules.

        Args:
            finger, gait, balance, sensor: Each (B, embed_dim).

        Returns:
            fused_emb (B, 512), attn_weights (B, 4, 4), dep_matrix (B, 4, 4),
            lag_weights (B, max_lag+1), contrastive_loss scalar.
        """
        tokens = torch.stack([finger, gait, balance, sensor], dim=1)
        attended, attn_weights = self.cross_limb_attn(tokens)
        graph_out = self.tdg(attended)
        dep_matrix = self.ddm(graph_out)
        lag_fused, lag_weights = self.time_lag(finger, gait)

        finger_e = graph_out[:, 0, :]
        gait_e = graph_out[:, 1, :] + lag_fused
        balance_e = graph_out[:, 2, :]
        sensor_e = graph_out[:, 3, :]

        contrastive_loss = (
            self.contrastive_loss_fn(finger_e, gait_e)
            if self.training else torch.tensor(0.0, device=finger.device)
        )

        fused_emb = self.fusion(torch.cat([finger_e, gait_e, balance_e, sensor_e], dim=-1))
        return fused_emb, attn_weights, dep_matrix, lag_weights, contrastive_loss
