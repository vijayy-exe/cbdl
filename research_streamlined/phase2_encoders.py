"""
CBDL Phase 2 — Modality Encoders, Training + Linear-Probe Sanity Checks, and
the Phase 7 finger-only baseline (consolidated research-release version).

Merges three originally-separate files from ../code/:
  - model.py                  the encoder architecture (WaveformAdapter,
                               FingerWaveformEncoder, MPowerMLPBranch/Decoder,
                               FusionLayer, GaitEncoder, LinearProbe,
                               CBDLPhase2Model) — unchanged.
  - train_probes.py            Phase 2 training loop + linear-probe sanity
                               checks against real labels (PADS primary,
                               Tappy/GaitRec secondary) — unchanged, just
                               renamed run -> run_phase2_training /
                               parse_args -> parse_args_phase2_training
                               since baseline_finger_only_f1.py defined its
                               own run()/parse_args() that would otherwise
                               collide with these in one namespace.
  - baseline_finger_only_f1.py  Phase 7 macro-F1 baseline computed on the
                               already-trained Phase 2 checkpoint (loads
                               phase2_checkpoint.pt — does not retrain
                               anything) — unchanged, renamed the same way
                               (run -> run_baseline_finger).

See ../code/ for the original, unmerged scripts and
../CBDL_PROJECT_DOCUMENTATION.md §3.3 / §5.6 for the architecture and result
writeups.

Usage:
    python phase2_encoders.py train    --data-root "." --epochs 15   # trains model.py's encoders, saves phase2_checkpoint.pt
    python phase2_encoders.py baseline --data-root "."                # loads that checkpoint, prints the finger-only macro-F1 baseline
Run `python phase2_encoders.py <task> --help` for that task's full flag list.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# MODEL ARCHITECTURE (from model.py)
# ============================================================================

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


# ============================================================================
# PHASE 2 TRAINING + PROBES (from train_probes.py)
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Datasets ────────────────────────────────────────────────────────────────


class WaveformDataset(Dataset):
    def __init__(self, waveforms: np.ndarray, masks: np.ndarray, labels: np.ndarray):
        self.waveforms = waveforms
        self.masks = masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.waveforms[idx]).float(),
            torch.from_numpy(self.masks[idx]).bool(),
            int(self.labels[idx]),
        )


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.features[idx]).float()


def load_pads(root: Path):
    w = np.load(root / "pads_preprocessed" / "pads_waveforms.npy")
    m = np.load(root / "pads_preprocessed" / "pads_masks.npy")
    labels = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv")
    y = labels["label"].to_numpy()
    split = labels["split"].to_numpy()
    return w, m, y, split, {0: "Healthy", 1: "Parkinson's", 2: "Other Movement Disorders"}


def load_tappy(root: Path):
    w = np.load(root / "tappy_preprocessed" / "tappy_waveforms.npy")
    m = np.load(root / "tappy_preprocessed" / "tappy_masks.npy")
    labels = pd.read_csv(root / "tappy_preprocessed" / "tappy_labels.csv")
    y = labels["Parkinsons"].astype(int).to_numpy()
    split = labels["Split"].to_numpy()
    return w, m, y, split, {0: "No PD", 1: "PD"}


def load_gaitrec(root: Path):
    w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")
    classes = sorted(labels["ClassLabel"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = labels["ClassLabel"].map(class_to_idx).to_numpy()
    split = labels["Split"].to_numpy()
    return w, y, split, {i: c for c, i in class_to_idx.items()}


def load_mpower(root: Path):
    f = np.load(root / "mpower_preprocessed" / "mpower_features.npy")
    labels = pd.read_csv(root / "mpower_preprocessed" / "mpower_labels.csv")
    split = labels["Split"].to_numpy()
    return f, split


def majority_baseline(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()


# ─── Training ────────────────────────────────────────────────────────────────


def run_phase2_training(args) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    root = Path(args.data_root)
    pads_w, pads_m, pads_y, pads_split, pads_classes = load_pads(root)
    tappy_w, tappy_m, tappy_y, tappy_split, tappy_classes = load_tappy(root)
    gait_w, gait_y, gait_split, gait_classes = load_gaitrec(root)
    mpower_f, mpower_split = load_mpower(root)

    print(f"PADS:     waveforms {pads_w.shape}, classes {pads_classes}")
    print(f"Tappy:    waveforms {tappy_w.shape}, classes {tappy_classes}")
    print(f"GaitRec:  waveforms {gait_w.shape}, classes {gait_classes}")
    print(f"mPower:   features  {mpower_f.shape} (unlabeled, autoencoder-only)")

    def loaders_for(w, m, y, split, batch_size):
        out = {}
        for s in ["train", "val", "test"]:
            mask_s = split == s
            ds = WaveformDataset(w[mask_s], m[mask_s], y[mask_s])
            out[s] = DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"))
        return out

    pads_loaders = loaders_for(pads_w, pads_m, pads_y, pads_split, args.batch_size_small)
    tappy_loaders = loaders_for(tappy_w, tappy_m, tappy_y, tappy_split, args.batch_size)
    gait_loaders = {
        s: DataLoader(
            WaveformDataset(gait_w[gait_split == s], np.ones((( gait_split == s).sum(), gait_w.shape[2]), dtype=bool), gait_y[gait_split == s]),
            batch_size=args.batch_size, shuffle=(s == "train"),
        )
        for s in ["train", "val", "test"]
    }
    mpower_loaders = {
        s: DataLoader(TabularDataset(mpower_f[mpower_split == s]), batch_size=args.batch_size, shuffle=(s == "train"))
        for s in ["train", "val", "test"]
    }

    model = CBDLPhase2Model().to(device)
    probe_pads = LinearProbe(128, len(pads_classes)).to(device)
    probe_pads_waveform_only = LinearProbe(128, len(pads_classes)).to(device)
    probe_tappy = LinearProbe(128, len(tappy_classes)).to(device)
    probe_gait = LinearProbe(128, len(gait_classes)).to(device)

    print(f"\nModel parameters: {model.n_params():,}")
    print(f"PADS majority baseline (test): {majority_baseline(pads_y[pads_split=='test']):.3f}")
    print(f"Tappy majority baseline (test): {majority_baseline(tappy_y[tappy_split=='test']):.3f}")
    print(f"GaitRec majority baseline (test): {majority_baseline(gait_y[gait_split=='test']):.3f}")

    all_params = (
        list(model.parameters())
        + list(probe_pads.parameters())
        + list(probe_pads_waveform_only.parameters())
        + list(probe_tappy.parameters())
        + list(probe_gait.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    mse = nn.MSELoss()

    def class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
        """Inverse-frequency weights, fit on TRAIN only — fixes the imbalance that was
        letting the model coast toward the majority class instead of learning real signal."""
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        w = counts.sum() / (n_classes * counts)
        return torch.tensor(w, dtype=torch.float32, device=device)

    ce_pads = nn.CrossEntropyLoss(weight=class_weights(pads_y[pads_split == "train"], len(pads_classes)))
    ce_tappy = nn.CrossEntropyLoss(weight=class_weights(tappy_y[tappy_split == "train"], len(tappy_classes)))
    ce_gait = nn.CrossEntropyLoss(weight=class_weights(gait_y[gait_split == "train"], len(gait_classes)))
    print(f"PADS class weights:  {ce_pads.weight.tolist()}")
    print(f"Tappy class weights: {ce_tappy.weight.tolist()}")

    # PADS has 469 samples vs Tappy's 23,752 sharing the same trunk — without
    # oversampling, ~742 Tappy batches/epoch swamp ~11 PADS batches, diluting
    # PADS's gradient signal in the shared trunk. Repeat PADS's train loader to
    # roughly match Tappy's step count so both get comparable trunk influence.
    pads_repeat_factor = max(1, round(len(tappy_loaders["train"]) / max(1, len(pads_loaders["train"]))))
    print(f"PADS train-loader repeat factor this epoch: {pads_repeat_factor}x "
          f"({len(pads_loaders['train'])} batches -> ~{len(pads_loaders['train']) * pads_repeat_factor})")

    GRAD_CLIP_NORM = 5.0

    def run_epoch(split: str, train: bool):
        model.train(train)
        probe_pads.train(train)
        probe_pads_waveform_only.train(train)
        probe_tappy.train(train)
        probe_gait.train(train)

        stats = {"pads_acc": [], "pads_wf_acc": [], "tappy_acc": [], "gait_acc": [], "mpower_recon_loss": []}

        # PADS — oversampled during training only, to counter the ~50x size
        # mismatch against Tappy on the shared trunk (see note above).
        n_repeats = pads_repeat_factor if train else 1
        for _ in range(n_repeats):
            for x, mask, y in pads_loaders[split]:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                with torch.set_grad_enabled(train):
                    wf_embed = model.encode_finger_waveform(x, mask, source="pads")
                    fused = model.encode_finger_fused(wf_embed, None, x.size(0), device)
                    logits_fused = probe_pads(fused)
                    logits_wf = probe_pads_waveform_only(wf_embed)
                    loss = ce_pads(logits_fused, y) + ce_pads(logits_wf, y)
                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP_NORM)
                        optimizer.step()
                stats["pads_acc"].append(accuracy(logits_fused, y))
                stats["pads_wf_acc"].append(accuracy(logits_wf, y))

        # Tappy
        for x, mask, y in tappy_loaders[split]:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            with torch.set_grad_enabled(train):
                wf_embed = model.encode_finger_waveform(x, mask, source="tappy")
                fused = model.encode_finger_fused(wf_embed, None, x.size(0), device)
                logits = probe_tappy(fused)
                loss = ce_tappy(logits, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP_NORM)
                    optimizer.step()
            stats["tappy_acc"].append(accuracy(logits, y))

        # GaitRec
        for x, mask, y in gait_loaders[split]:
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(train):
                gait_embed = model.encode_gait(x)
                logits = probe_gait(gait_embed)
                loss = ce_gait(logits, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP_NORM)
                    optimizer.step()
            stats["gait_acc"].append(accuracy(logits, y))

        # mPower (autoencoder — no label)
        for feats in mpower_loaders[split]:
            feats = feats.to(device)
            with torch.set_grad_enabled(train):
                z = model.mpower_branch(feats)
                recon = model.mpower_decoder(z)
                loss = mse(recon, feats)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP_NORM)
                    optimizer.step()
            stats["mpower_recon_loss"].append(loss.item())

        return {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch("train", train=True)
        val_stats = run_epoch("val", train=False)
        print(
            f"Epoch {epoch:2d} | "
            f"PADS(fused/wf) {train_stats['pads_acc']:.3f}/{train_stats['pads_wf_acc']:.3f} -> "
            f"val {val_stats['pads_acc']:.3f}/{val_stats['pads_wf_acc']:.3f} | "
            f"Tappy {train_stats['tappy_acc']:.3f} -> val {val_stats['tappy_acc']:.3f} | "
            f"Gait {train_stats['gait_acc']:.3f} -> val {val_stats['gait_acc']:.3f} | "
            f"mPower recon {train_stats['mpower_recon_loss']:.4f} -> val {val_stats['mpower_recon_loss']:.4f}"
        )

    test_stats = run_epoch("test", train=False)

    print("\n" + "=" * 70)
    print("PHASE 2 SANITY-CHECK RESULTS (test split)")
    print("=" * 70)
    print(f"PADS finger probe (fused)        : {test_stats['pads_acc']:.3f}  (majority baseline {majority_baseline(pads_y[pads_split=='test']):.3f})")
    print(f"PADS finger probe (waveform-only): {test_stats['pads_wf_acc']:.3f}  (majority baseline {majority_baseline(pads_y[pads_split=='test']):.3f})")
    print(f"Tappy finger probe (secondary)   : {test_stats['tappy_acc']:.3f}  (majority baseline {majority_baseline(tappy_y[tappy_split=='test']):.3f})")
    print(f"GaitRec gait probe               : {test_stats['gait_acc']:.3f}  (majority baseline {majority_baseline(gait_y[gait_split=='test']):.3f})")
    print(f"mPower reconstruction MSE        : {test_stats['mpower_recon_loss']:.4f}  (unsupervised — no baseline)")
    print("=" * 70)
    print(
        "\nLIMITATION (state in paper): GaitRec/Tappy/PADS/mPower are disjoint subject\n"
        "pools. The PADS 'fused' probe above uses the fusion layer's learned\n"
        "'absent mPower' token, NOT real mPower features for that sample — no sample\n"
        "has both. mPower's MLP branch is trained separately via feature\n"
        "reconstruction, since no diagnostic label co-occurs with its features in the\n"
        "data on hand. The fused-vs-waveform-only comparison above therefore tests\n"
        "whether the extra fusion projection (with a constant learned token) helps or\n"
        "hurts — not a real mPower-contribution ablation. A genuine mPower-fusion\n"
        "ablation needs a source dataset with per-sample mPower + diagnostic label.\n"
    )

    if test_stats["pads_acc"] <= majority_baseline(pads_y[pads_split == "test"]):
        print("WARNING: PADS probe did not beat majority baseline — per the CBDL plan, "
              "stop and debug before building the Cross-Body Dependency Module.")
    else:
        print("PADS probe beats majority baseline — proceed to Phase 3 (Cross-Body Dependency Module).")

    torch.save(
        {
            "model": model.state_dict(),
            "probe_pads": probe_pads.state_dict(),
            "probe_pads_waveform_only": probe_pads_waveform_only.state_dict(),
            "probe_tappy": probe_tappy.state_dict(),
            "probe_gait": probe_gait.state_dict(),
        },
        root / "phase2_checkpoint.pt",
    )
    print(f"\nSaved checkpoint to {root / 'phase2_checkpoint.pt'}")


def parse_args_phase2_training():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--batch-size-small", type=int, default=32, help="PADS has only 469 samples total")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ============================================================================
# PHASE 7 FINGER-ONLY BASELINE (from baseline_finger_only_f1.py)
# ============================================================================

def run_baseline_finger(args) -> None:
    device = get_device()
    root = Path(args.data_root)

    pads_w, pads_m, pads_y, pads_split, pads_classes = load_pads(root)

    model = CBDLPhase2Model().to(device)
    ckpt = torch.load(root / "phase2_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    probe_fused = LinearProbe(128, 3).to(device)
    probe_fused.load_state_dict(ckpt["probe_pads"])
    probe_wf = LinearProbe(128, 3).to(device)
    probe_wf.load_state_dict(ckpt["probe_pads_waveform_only"])
    model.eval(); probe_fused.eval(); probe_wf.eval()

    test_idx = np.where(pads_split == "test")[0]
    x = torch.from_numpy(pads_w[test_idx]).float().to(device)
    m = torch.from_numpy(pads_m[test_idx]).bool().to(device)
    y = pads_y[test_idx]

    with torch.no_grad():
        wf_embed = model.encode_finger_waveform(x, m, source="pads")
        fused = model.encode_finger_fused(wf_embed, None, x.size(0), device)
        preds_fused = probe_fused(fused).argmax(-1).cpu().numpy()
        preds_wf = probe_wf(wf_embed).argmax(-1).cpu().numpy()

    print("=" * 70)
    print("PHASE 2 FINGER-ONLY BASELINE — full test set (71 samples), macro-F1")
    print("=" * 70)
    print(f"Waveform-only (no mPower fusion): acc={(preds_wf==y).mean():.3f}  macro-F1={f1_score(y, preds_wf, average='macro'):.3f}")
    print(f"Fused (with absent-mPower token):  acc={(preds_fused==y).mean():.3f}  macro-F1={f1_score(y, preds_fused, average='macro'):.3f}")
    print(f"Majority baseline: {majority_baseline(y):.3f}")
    print("=" * 70)


def parse_args_baseline_finger():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────
# Unified CLI — dispatches to Phase 2 training or the Phase 7 finger-only
# baseline (which just evaluates the checkpoint training already produced).
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    tasks = {"train": (run_phase2_training, parse_args_phase2_training),
              "baseline": (run_baseline_finger, parse_args_baseline_finger)}
    if len(sys.argv) < 2 or sys.argv[1] not in tasks:
        print(f"usage: {sys.argv[0]} {{{','.join(tasks)}}} [task-specific args...]", file=sys.stderr)
        print(f"Run '{sys.argv[0]} <task> --help' to see that task's arguments.", file=sys.stderr)
        sys.exit(1)
    task = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    run_fn, parse_fn = tasks[task]
    run_fn(parse_fn())


if __name__ == "__main__":
    main()
