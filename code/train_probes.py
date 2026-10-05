"""
CBDL Phase 2 — training + linear-probe sanity checks.

Trains, in one run:
  1. Finger Waveform Encoder (shared BiGRU trunk, Tappy+PADS adapters),
     probed against PADS's real diagnostic label (PD/HC/Other) — the
     PRIMARY sanity check per the CBDL plan. If this can't beat the
     majority-class baseline, stop and debug before building anything on
     top (Phase 2 instruction).
  2. The same shared trunk, probed against Tappy's own Parkinsons label —
     a SECONDARY check that the shared trunk carries cross-source signal.
  3. Gait Encoder (CNN-LSTM), probed against GaitRec's pathology label
     (HC/A/C/H/K) — secondary gait-side check.
  4. mPower's MLP branch, trained via feature-reconstruction (autoencoder)
     since no diagnostic label co-occurs with its features in the data on
     hand (see model.py docstring for why).
  5. An ablation on PADS: probe accuracy from the waveform embedding alone
     vs. from the fused Finger Embedding (waveform + learned "absent
     mPower" token) — isolates whether the fusion layer itself helps or
     hurts, given no real per-sample mPower pairing exists (see Limitations
     note printed at the end).

Usage:
    python train_probes.py --data-root "." --epochs 15
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import CBDLPhase2Model, LinearProbe


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


def run(args) -> None:
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--batch-size-small", type=int, default=32, help="PADS has only 469 samples total")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
