"""
Stage 7 — FiLM-Style Personalization for the CBDL Pipeline.

Personalizes the 512-d cross-body embedding using per-subject metadata
via Feature-wise Linear Modulation (FiLM): output = γ * embedding + β.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class FiLMPersonalization(nn.Module):
    """
    FiLM conditioning module.

    Maps 4-d metadata [age_norm, gender, dominant_hand, baseline_norm]
    to per-channel scale (γ) and shift (β) for the 512-d cross-body embedding.

    Args:
        meta_dim: Metadata vector dimension (default 4).
        embed_dim: Cross-body embedding dimension (default 512).
        hidden_dim: Hidden dimension of the conditioning MLP.
    """

    def __init__(self, meta_dim: int = 4, embed_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cross_body_emb: torch.Tensor, metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply FiLM conditioning.

        Args:
            cross_body_emb: (B, 512) embedding.
            metadata: (B, 4) with [age_norm, gender, dominant_hand, baseline_norm].

        Returns:
            personalized_emb (B, 512), gamma (B, 512), beta (B, 512).
        """
        film_params = self.mlp(metadata)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma + 1.0
        return gamma * cross_body_emb + beta, gamma, beta


def prepare_metadata_tensor(
    age: float,
    gender: float,
    dominant_hand: float,
    baseline_profile: "np.ndarray",
    age_min: float = 40.0,
    age_max: float = 80.0,
) -> torch.Tensor:
    """
    Normalize per-subject metadata to a 4-d tensor.

    Args:
        age: Raw age in years (40–80).
        gender: 0.0 (male) or 1.0 (female).
        dominant_hand: 0.0 (right) or 1.0 (left).
        baseline_profile: Numpy array summarized as its mean.
        age_min: Min age for normalization.
        age_max: Max age for normalization.

    Returns:
        Tensor of shape (4,): [age_norm, gender, dominant_hand, baseline_norm].
    """
    age_norm = (age - age_min) / (age_max - age_min + 1e-8)
    baseline_norm = float(np.mean(baseline_profile))
    return torch.tensor([age_norm, gender, dominant_hand, baseline_norm], dtype=torch.float32)
