"""
CBDL Phase 2 — Modality Encoders (per CBDL_Development_Plan.md Phase 2).

Components:
  - WaveformAdapter        per-source linear/1D-conv projection into a
                            common embedding width (one per Tappy/PADS).
  - FingerWaveformEncoder   shared BiGRU trunk fed by both adapters.
  - MPowerMLPBranch         small MLP over mPower's handcrafted tapFeatures.
  - MPowerDecoder           reconstruction head so the MLP branch gets a
                            training signal — see note below.
  - FusionLayer             concatenates waveform-trunk output with the
                            mPower branch output (or a learned "absent"
                            token when one modality has no paired sample —
                            see note) and projects to the final Finger
                            Embedding.
  - GaitEncoder             CNN-LSTM over GaitRec's 18-channel waveform.
  - LinearProbe             single nn.Linear sanity-check head.

Note on the fusion/mPower training signal (a real, disclosed limitation):
GaitRec, Tappy, PADS, and mPower are four disjoint subject pools (see the
preprocessing data cards) — no sample has both a waveform (Tappy/PADS) and
mPower features. So the Fusion Layer's per-plan design ("concatenate the
Finger Waveform Encoder's output with the mPower MLP branch's output") can
never see paired real data for a single sample. This module resolves that
with a learned "modality absent" embedding substituted for whichever side
has no data for a given sample (a standard missing-modality pattern), and
the mPower MLP branch is given its own training signal via a lightweight
feature-reconstruction (autoencoder) objective, since no diagnostic label
co-occurs with mPower's features in the data on hand. State this plainly in
the paper's Limitations, consistent with the population-level-pairing
framing already used for the later Cross-Body Dependency Module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_window_pool(seq: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Splits the time axis into k contiguous, (as-close-to-)equal segments and
    masked-mean-pools each — used by Phase 3's windowed encoder outputs so
    Lagged Cross-Attention has a sequence of k window embeddings to attend
    over, regardless of the source's native T (PADS T=2928, Tappy T=888,
    GaitRec T=101 all reduce to the same k).

    seq:  [B, T, D] float
    mask: [B, T]    bool (True = real timestep)
    Returns: [B, k, D] float
    """
    b, t, d = seq.shape
    bounds = torch.linspace(0, t, k + 1).round().long()
    windows = []
    mask_f = mask.unsqueeze(-1).float()
    for i in range(k):
        lo, hi = bounds[i].item(), max(bounds[i + 1].item(), bounds[i].item() + 1)
        seg = seq[:, lo:hi, :]
        seg_mask = mask_f[:, lo:hi, :]
        summed = (seg * seg_mask).sum(dim=1)
        counts = seg_mask.sum(dim=1).clamp(min=1.0)
        windows.append(summed / counts)
    return torch.stack(windows, dim=1)  # [B, k, D]


class WaveformAdapter(nn.Module):
    """Per-source projection: [B, C_in, T] -> [B, adapter_dim, T]."""

    def __init__(self, in_channels: int, adapter_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, adapter_dim, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))


class FingerWaveformEncoder(nn.Module):
    """
    Shared BiGRU trunk. Input: [B, adapter_dim, T] (already source-adapted).
    Output: [B, out_dim] fixed-size embedding, masked mean-pooled over time.
    """

    def __init__(self, adapter_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gru = nn.GRU(adapter_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, adapter_dim, T] -> [B, T, adapter_dim] for GRU
        x = x.transpose(1, 2)
        out, _ = self.gru(x)  # [B, T, 2*hidden_dim]
        mask_f = mask.unsqueeze(-1).float()  # [B, T, 1]
        summed = (out * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.proj(pooled)

    def forward_windows(self, x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
        """Phase 3: same GRU trunk, but returns k window embeddings [B, k, out_dim] instead of one pooled vector."""
        x = x.transpose(1, 2)
        out, _ = self.gru(x)          # [B, T, 2*hidden_dim]
        out = self.proj(out)          # [B, T, out_dim] — project per-timestep before windowing
        return masked_window_pool(out, mask, k)


class MPowerMLPBranch(nn.Module):
    """Small MLP over mPower's 41 handcrafted features -> [B, out_dim]."""

    def __init__(self, in_features: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MPowerDecoder(nn.Module):
    """Reconstructs the 41 input features from the MLP branch's embedding — see module docstring."""

    def __init__(self, embed_dim: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FusionLayer(nn.Module):
    """
    Concatenates a waveform-trunk embedding with an mPower-branch embedding
    and projects to the final Finger Embedding. Either side may be the
    learned "absent" token (see module docstring) when a sample has no data
    for that modality.
    """

    def __init__(self, waveform_dim: int, mpower_dim: int, out_dim: int):
        super().__init__()
        self.absent_waveform = nn.Parameter(torch.zeros(waveform_dim))
        self.absent_mpower = nn.Parameter(torch.zeros(mpower_dim))
        self.proj = nn.Linear(waveform_dim + mpower_dim, out_dim)

    def forward(self, waveform_embed: torch.Tensor | None, mpower_embed: torch.Tensor | None, batch_size: int, device) -> torch.Tensor:
        wf = waveform_embed if waveform_embed is not None else self.absent_waveform.unsqueeze(0).expand(batch_size, -1).to(device)
        mp = mpower_embed if mpower_embed is not None else self.absent_mpower.unsqueeze(0).expand(batch_size, -1).to(device)
        return self.proj(torch.cat([wf, mp], dim=-1))


class GaitEncoder(nn.Module):
    """CNN-LSTM over GaitRec's fixed [18, 101] waveform -> [B, out_dim]."""

    def __init__(self, in_channels: int, conv_dim: int, lstm_hidden: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim * 2, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_dim * 2, lstm_hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(lstm_hidden * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 18, T]
        h = self.conv(x)              # [B, conv_dim*2, T]
        h = h.transpose(1, 2)         # [B, T, conv_dim*2]
        out, _ = self.lstm(h)         # [B, T, 2*lstm_hidden]
        pooled = out.mean(dim=1)      # GaitRec trials have no padding — plain mean is fine
        return self.proj(pooled)

    def forward_windows(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Phase 3: same CNN-LSTM, but returns k window embeddings [B, k, out_dim]."""
        h = self.conv(x)
        h = h.transpose(1, 2)
        out, _ = self.lstm(h)
        out = self.proj(out)  # [B, T, out_dim]
        mask = torch.ones(out.shape[0], out.shape[1], dtype=torch.bool, device=out.device)  # GaitRec: no padding
        return masked_window_pool(out, mask, k)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CBDLPhase2Model(nn.Module):
    """Bundles every Phase 2 component. Encoders are small by design (few hundred K params total)."""

    def __init__(
        self,
        tappy_channels: int = 3,
        pads_channels: int = 6,
        gaitrec_channels: int = 18,
        mpower_features: int = 41,
        adapter_dim: int = 64,
        waveform_hidden: int = 64,
        waveform_out: int = 128,
        mpower_hidden: int = 64,
        mpower_out: int = 128,
        finger_embed_dim: int = 128,
        gait_conv_dim: int = 32,
        gait_lstm_hidden: int = 64,
        gait_embed_dim: int = 128,
    ):
        super().__init__()
        self.tappy_adapter = WaveformAdapter(tappy_channels, adapter_dim)
        self.pads_adapter = WaveformAdapter(pads_channels, adapter_dim)
        self.finger_trunk = FingerWaveformEncoder(adapter_dim, waveform_hidden, waveform_out)

        self.mpower_branch = MPowerMLPBranch(mpower_features, mpower_hidden, mpower_out)
        self.mpower_decoder = MPowerDecoder(mpower_out, mpower_hidden, mpower_features)

        self.fusion = FusionLayer(waveform_out, mpower_out, finger_embed_dim)

        self.gait_encoder = GaitEncoder(gaitrec_channels, gait_conv_dim, gait_lstm_hidden, gait_embed_dim)

        assert finger_embed_dim == gait_embed_dim, "Finger and Gait embeddings must match dim for later cross-attention (Phase 3)."

    def encode_finger_waveform(self, x: torch.Tensor, mask: torch.Tensor, source: str) -> torch.Tensor:
        adapter = self.tappy_adapter if source == "tappy" else self.pads_adapter
        return self.finger_trunk(adapter(x), mask)

    def encode_finger_fused(self, waveform_embed: torch.Tensor | None, mpower_embed: torch.Tensor | None, batch_size: int, device) -> torch.Tensor:
        return self.fusion(waveform_embed, mpower_embed, batch_size, device)

    def encode_gait(self, x: torch.Tensor) -> torch.Tensor:
        return self.gait_encoder(x)

    def encode_finger_windows(self, x: torch.Tensor, mask: torch.Tensor, source: str, k: int) -> torch.Tensor:
        adapter = self.tappy_adapter if source == "tappy" else self.pads_adapter
        return self.finger_trunk.forward_windows(adapter(x), mask, k)

    def encode_gait_windows(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return self.gait_encoder.forward_windows(x, k)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
