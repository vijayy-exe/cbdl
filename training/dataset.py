"""
Training utilities and dataset classes for the CBDL pipeline.

Provides a PyTorch Dataset over preprocessed windows, and training loops
for the encoders, cross-body module, and representation module.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CBDLWindowDataset(Dataset):
    """
    PyTorch Dataset over preprocessed CBDL windows.

    Each item returns one window across all streams plus subject metadata.

    Args:
        processed_dir: Path to directory of .npz processed files.
        subject_ids: List of subject IDs to include. If None, use all.
    """

    def __init__(
        self,
        processed_dir: str,
        subject_ids: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        files = sorted(self.processed_dir.glob("*.npz"))
        if subject_ids is not None:
            files = [f for f in files if int(f.stem.split("_")[1]) in subject_ids]
        self.files = files
        self._index: List[Tuple[int, int]] = []  # (file_idx, window_idx)
        self._cache: Dict[int, dict] = {}

        for fi, fpath in enumerate(self.files):
            data = np.load(fpath, allow_pickle=True)
            n_windows = data["finger"].shape[0]
            for wi in range(n_windows):
                self._index.append((fi, wi))
            self._cache[fi] = dict(data)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return one window as a dict of stream tensors and metadata.

        Args:
            idx: Index into the flat window index.

        Returns:
            Dict with keys: finger, gait, insole, phone, wrist, physio,
            metadata, subject_id, label.
        """
        fi, wi = self._index[idx]
        data = self._cache[fi]

        def _get(key: str) -> torch.Tensor:
            arr = data[key][wi].astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            return torch.from_numpy(arr)

        meta_arr = np.array([
            float(data["meta_age"]),
            float(data["meta_gender"]),
            float(data["meta_dominant_hand"]),
            float(np.mean(data["meta_baseline_profile"])),
        ], dtype=np.float32)

        # Normalize age to [0,1]
        meta_arr[0] = (meta_arr[0] - 40.0) / 40.0

        return {
            "finger": _get("finger"),
            "gait": _get("gait"),
            "insole": _get("insole_pressure"),
            "phone": _get("phone_acc_gyro"),
            "wrist": _get("wrist"),
            "physio": _get("physio"),
            "metadata": torch.from_numpy(meta_arr),
            "subject_id": int(data["meta_subject_id"]),
            "label": int(data["meta_label"]),
            "updrs": float(data["meta_updrs"]),
            "fall_risk": float(data["meta_fall_risk"]),
            "motor_severity": float(data["meta_motor_severity"]),
        }


def make_dataloader(
    processed_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
    subject_ids: Optional[List[int]] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader over the CBDL window dataset.

    Args:
        processed_dir: Directory with processed .npz files.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        subject_ids: Optional list of subject IDs to include.
        num_workers: DataLoader worker count.

    Returns:
        PyTorch DataLoader.
    """
    dataset = CBDLWindowDataset(processed_dir, subject_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
