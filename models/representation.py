"""
Stage 5 — Disease-Agnostic Neuromotor Representation.

Self-supervised pretraining with:
  1. Masked reconstruction (25% of timesteps, Transformer decoder)
  2. VICReg loss (variance + invariance + covariance)

No disease labels used during pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MaskedReconstructionHead(nn.Module):
    """
    Reconstructs masked timestep features from the cross-body embedding.

    Args:
        cross_body_dim: Dimensionality of cross-body embedding (512).
        n_channels: Number of sensor channels to reconstruct.
        window_size: Number of timesteps per window.
        mask_ratio: Fraction of timesteps to mask.
        n_decoder_layers: Number of Transformer decoder layers.
        n_heads: Decoder attention heads.
    """

    def __init__(self, cross_body_dim: int = 512, n_channels: int = 6, window_size: int = 200, mask_ratio: float = 0.25, n_decoder_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.window_size = window_size
        self.n_channels = n_channels
        self.memory_proj = nn.Linear(cross_body_dim, 128)
        self.pos_emb = nn.Embedding(window_size, 128)
        self.input_proj = nn.Linear(n_channels, 128)
        decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=n_heads, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.out_proj = nn.Linear(128, n_channels)

    def forward(self, x: torch.Tensor, cross_body_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply masking and reconstruct.

        Args:
            x: (B, C, T) signal tensor.
            cross_body_emb: (B, 512) cross-body embedding.

        Returns:
            reconstruction (B, T, C), mask (B, T), reconstruction_loss scalar.
        """
        B, C, T = x.shape
        x_seq = x.permute(0, 2, 1)
        mask = torch.rand(B, T, device=x.device) < self.mask_ratio
        x_masked = x_seq.clone()
        x_masked[mask] = 0.0
        inp = self.input_proj(x_masked)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        inp = inp + self.pos_emb(positions)
        memory = self.memory_proj(cross_body_emb).unsqueeze(1)
        decoded = self.decoder(inp, memory)
        reconstruction = self.out_proj(decoded)
        target = x_seq.detach()
        recon_loss = F.mse_loss(reconstruction[mask], target[mask]) if mask.any() else torch.tensor(0.0, device=x.device)
        return reconstruction, mask, recon_loss


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Return off-diagonal elements of a square matrix as a 1-D tensor."""
    D = matrix.shape[0]
    return matrix.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """
    VICReg loss: Variance + Invariance + Covariance.

    Args:
        lambda_inv: Invariance weight.
        mu_var: Variance weight.
        nu_cov: Covariance weight.
        eps: Numerical stability epsilon.
    """

    def __init__(self, lambda_inv: float = 25.0, mu_var: float = 25.0, nu_cov: float = 1.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.lambda_inv = lambda_inv
        self.mu_var = mu_var
        self.nu_cov = nu_cov
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z1, z2: Embeddings of shape (B, D).
        Returns:
            total_loss scalar, dict of individual term values.
        """
        B, D = z1.shape
        inv_loss = F.mse_loss(z1, z2)
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        var_loss = (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean()) * 0.5
        z1c = z1 - z1.mean(dim=0)
        z2c = z2 - z2.mean(dim=0)
        cov_z1 = (z1c.T @ z1c) / (B - 1)
        cov_z2 = (z2c.T @ z2c) / (B - 1)
        cov_loss = (_off_diagonal(cov_z1).pow(2).sum() + _off_diagonal(cov_z2).pow(2).sum()) / D
        total = self.lambda_inv * inv_loss + self.mu_var * var_loss + self.nu_cov * cov_loss
        return total, {"invariance": inv_loss.item(), "variance": var_loss.item(), "covariance": cov_loss.item()}


class NeuromotorRepresentation(nn.Module):
    """
    Stage 5 self-supervised pretraining head.

    Combines masked reconstruction and VICReg to learn disease-agnostic embeddings.

    Args:
        cross_body_dim: Cross-body embedding dimension (512).
        n_channels: Sensor channels to reconstruct.
        window_size: Window length in samples.
        mask_ratio: Fraction of timesteps masked.
        vicreg_lambda/mu/nu: VICReg loss weights.
    """

    def __init__(self, cross_body_dim: int = 512, n_channels: int = 6, window_size: int = 200, mask_ratio: float = 0.25, vicreg_lambda: float = 25.0, vicreg_mu: float = 25.0, vicreg_nu: float = 1.0) -> None:
        super().__init__()
        self.recon_head = MaskedReconstructionHead(cross_body_dim, n_channels, window_size, mask_ratio)
        self.vicreg = VICRegLoss(vicreg_lambda, vicreg_mu, vicreg_nu)
        self.projector = nn.Sequential(
            nn.Linear(cross_body_dim, cross_body_dim),
            nn.BatchNorm1d(cross_body_dim),
            nn.GELU(),
            nn.Linear(cross_body_dim, cross_body_dim),
        )

    def forward(self, finger_stream: torch.Tensor, cross_body_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute pretraining losses.

        Args:
            finger_stream: (B, 6, T) finger windows for reconstruction.
            cross_body_emb: (B, 512) cross-body embedding.

        Returns:
            total_loss, cross_body_emb (pass-through), loss_terms dict.
        """
        _, _, recon_loss = self.recon_head(finger_stream, cross_body_emb)
        if self.training:
            z1 = self.projector(cross_body_emb + torch.randn_like(cross_body_emb) * 0.01)
            z2 = self.projector(cross_body_emb + torch.randn_like(cross_body_emb) * 0.01)
            vicreg_loss, vicreg_terms = self.vicreg(z1, z2)
        else:
            vicreg_loss = torch.tensor(0.0, device=cross_body_emb.device)
            vicreg_terms = {"invariance": 0.0, "variance": 0.0, "covariance": 0.0}
        total_loss = recon_loss + vicreg_loss
        loss_terms = {
            "reconstruction": recon_loss.item(),
            "vicreg_total": vicreg_loss.item() if isinstance(vicreg_loss, torch.Tensor) else vicreg_loss,
            **{f"vicreg_{k}": v for k, v in vicreg_terms.items()},
        }
        return total_loss, cross_body_emb, loss_terms
