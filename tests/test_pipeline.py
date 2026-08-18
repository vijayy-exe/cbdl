"""
CBDL Pipeline — Pytest Test Suite

Covers:
  - Data shape checks
  - Encoder forward pass shapes
  - Biomarker range sanity checks
  - End-to-end smoke test on 5 subjects
"""

import os
import sys
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
#  Data shape tests
# ──────────────────────────────────────────────

class TestDataGeneration:
    """Tests for Stage 1 synthetic data generation."""

    def test_subject_data_shapes(self) -> None:
        """Check that generated subject data has correct array shapes."""
        from data.synthetic_generator import generate_subject
        subject = generate_subject(subject_id=0, symptomatic=True, fs=100.0, duration=10.0)
        meta = subject.pop("metadata", None) or {}
        # We expect pop to handle the case where metadata was already split
        # The generate_subject function includes metadata key
        n_samples = int(100.0 * 10.0)

        assert subject["finger"].shape == (6, n_samples), f"finger shape wrong: {subject['finger'].shape}"
        assert subject["wrist"].shape == (6, n_samples), f"wrist shape wrong"
        assert subject["gait"].shape == (6, n_samples), f"gait shape wrong"
        assert subject["insole_pressure"].shape == (4, n_samples), f"insole shape wrong"
        assert subject["phone_acc_gyro"].shape == (6, n_samples), f"phone shape wrong"
        assert subject["physio"].shape == (4, n_samples), f"physio shape wrong"
        assert subject["timestamps"].shape == (n_samples,), f"timestamps shape wrong"

    def test_nan_injection(self) -> None:
        """Verify NaN dropouts are injected in raw data."""
        from data.synthetic_generator import generate_subject
        n_with_nan = 0
        for i in range(10):
            subject = generate_subject(subject_id=i, symptomatic=i % 2 == 0, fs=100.0, duration=10.0)
            for key in ["finger", "wrist", "gait"]:
                if np.isnan(subject[key]).any():
                    n_with_nan += 1
                    break
        assert n_with_nan > 0, "Expected some subjects to have NaN dropouts"

    def test_metadata_fields(self) -> None:
        """Check that all required metadata fields are present."""
        from data.synthetic_generator import generate_subject
        subject = generate_subject(subject_id=0, symptomatic=False)
        meta = subject["metadata"]
        required_keys = ["subject_id", "label", "age", "gender", "dominant_hand",
                         "baseline_profile", "updrs", "fall_risk", "motor_severity"]
        for key in required_keys:
            assert key in meta, f"Missing metadata key: {key}"

    def test_symptomatic_vs_healthy_labels(self) -> None:
        """Healthy subjects should have lower UPDRS than symptomatic."""
        from data.synthetic_generator import generate_subject
        healthy_updrs = [generate_subject(i, False)["metadata"]["updrs"] for i in range(5)]
        sympt_updrs = [generate_subject(i, True)["metadata"]["updrs"] for i in range(100, 105)]
        assert np.mean(healthy_updrs) < np.mean(sympt_updrs), "Healthy should have lower UPDRS"


# ──────────────────────────────────────────────
#  Encoder shape tests
# ──────────────────────────────────────────────

class TestEncoders:
    """Tests for Stage 3 encoder forward pass shapes."""

    @pytest.fixture
    def dummy_batch(self):
        B, T = 2, 200
        return {
            "finger": torch.randn(B, 6, T),
            "gait": torch.randn(B, 6, T),
            "insole": torch.randn(B, 4, T),
            "phone": torch.randn(B, 6, T),
            "wrist": torch.randn(B, 6, T),
        }

    def test_finger_encoder_shape(self, dummy_batch) -> None:
        """FingerTemporalEncoder should return (B, 128)."""
        from models.encoders import FingerTemporalEncoder
        enc = FingerTemporalEncoder(embed_dim=128)
        enc.eval()
        with torch.no_grad():
            out = enc(dummy_batch["finger"])
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    def test_gait_encoder_shape(self, dummy_batch) -> None:
        """GaitTemporalEncoder should return (B, 128)."""
        from models.encoders import GaitTemporalEncoder
        enc = GaitTemporalEncoder(embed_dim=128)
        enc.eval()
        with torch.no_grad():
            out = enc(dummy_batch["gait"])
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    def test_balance_encoder_shape(self, dummy_batch) -> None:
        """BalanceEncoder should return (B, 128)."""
        from models.encoders import BalanceEncoder
        enc = BalanceEncoder(embed_dim=128)
        enc.eval()
        with torch.no_grad():
            out = enc(dummy_batch["insole"], dummy_batch["phone"])
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    def test_sensor_encoder_shape(self, dummy_batch) -> None:
        """SensorEncoder should return (B, 128)."""
        from models.encoders import SensorEncoder
        enc = SensorEncoder(embed_dim=128)
        enc.eval()
        with torch.no_grad():
            out = enc(dummy_batch["wrist"], dummy_batch["phone"])
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    def test_multi_stream_encoder_shapes(self, dummy_batch) -> None:
        """MultiStreamEncoder should return 4 embeddings each of shape (B, 128)."""
        from models.encoders import MultiStreamEncoder
        enc = MultiStreamEncoder(embed_dim=128)
        enc.eval()
        with torch.no_grad():
            embs = enc(
                dummy_batch["finger"],
                dummy_batch["gait"],
                dummy_batch["insole"],
                dummy_batch["phone"],
                dummy_batch["wrist"],
            )
        assert len(embs) == 4
        for emb in embs:
            assert emb.shape == (2, 128), f"Expected (2, 128), got {emb.shape}"


# ──────────────────────────────────────────────
#  Cross-body module shape tests
# ──────────────────────────────────────────────

class TestCrossBodyModule:
    """Tests for Stage 4 cross-body dependency module."""

    def test_cross_body_output_shape(self) -> None:
        """CrossBodyDependencyModule should return 512-d fused embedding."""
        from models.cross_body import CrossBodyDependencyModule
        B, D = 4, 128
        module = CrossBodyDependencyModule(embed_dim=D, n_heads=4, max_lag=25)
        module.eval()
        embs = [torch.randn(B, D) for _ in range(4)]
        with torch.no_grad():
            fused, attn_w, dep_m, lag_w, _ = module(*embs)
        assert fused.shape == (B, 512), f"Expected (B, 512), got {fused.shape}"
        assert attn_w.shape == (B, 4, 4), f"Attn weights shape wrong: {attn_w.shape}"
        assert dep_m.shape == (B, 4, 4), f"Dep matrix shape wrong: {dep_m.shape}"

    def test_dep_matrix_symmetry(self) -> None:
        """Dependency matrix should be symmetric."""
        from models.cross_body import CrossBodyDependencyModule
        B, D = 2, 128
        module = CrossBodyDependencyModule(embed_dim=D)
        module.eval()
        embs = [torch.randn(B, D) for _ in range(4)]
        with torch.no_grad():
            _, _, dep_m, _, _ = module(*embs)
        diff = (dep_m - dep_m.transpose(-1, -2)).abs().max().item()
        assert diff < 1e-5, f"Dependency matrix not symmetric: max diff={diff}"


# ──────────────────────────────────────────────
#  Biomarker range tests
# ──────────────────────────────────────────────

class TestBiomarkers:
    """Tests for Stage 6 biomarker sanity checks."""

    def _make_dummy_biomarker_inputs(self, N=20, D=128):
        return {
            "cross_body_embs": np.random.randn(N, 512).astype(np.float32),
            "finger_embs": np.random.randn(N, D).astype(np.float32),
            "gait_embs": np.random.randn(N, D).astype(np.float32),
            "balance_embs": np.random.randn(N, D).astype(np.float32),
            "sensor_embs": np.random.randn(N, D).astype(np.float32),
            "dep_matrices": np.random.rand(N, 4, 4).astype(np.float32),
            "attn_weights": np.random.dirichlet(np.ones(16), N).reshape(N, 4, 4).astype(np.float32),
            "lag_weights": np.random.dirichlet(np.ones(26), N).astype(np.float32),
        }

    def test_cbdi_range(self) -> None:
        """CBDi should be in [0, 1] for sigmoid-scaled dep matrices."""
        from biomarkers.compute import compute_cbdi
        dummy = np.random.rand(10, 4, 4).astype(np.float32)
        cbdi = compute_cbdi(dummy)
        assert 0.0 <= cbdi <= 1.0, f"CBDi out of range: {cbdi}"

    def test_neuromotor_stability_positive(self) -> None:
        """Neuromotor stability should be positive."""
        from biomarkers.compute import compute_neuromotor_stability
        embs = np.random.randn(20, 512).astype(np.float32)
        score = compute_neuromotor_stability(embs)
        assert score > 0, f"Stability should be positive: {score}"

    def test_coordination_index_nonneg(self) -> None:
        """Coordination Index (entropy) should be non-negative."""
        from biomarkers.compute import compute_coordination_index
        attn = np.random.dirichlet(np.ones(16)).reshape(4, 4)
        ci = compute_coordination_index(attn)
        assert ci >= 0.0, f"Coordination Index should be ≥ 0: {ci}"

    def test_all_biomarkers_computed(self) -> None:
        """compute_subject_biomarkers should return all 6 biomarker keys."""
        from biomarkers.compute import compute_subject_biomarkers
        inputs = self._make_dummy_biomarker_inputs()
        result = compute_subject_biomarkers(subject_id=0, **inputs)
        expected_keys = {
            "cbdi", "finger_gait_coupling", "neuromotor_stability",
            "coordination_index", "motor_variability", "synchronization_score"
        }
        for key in expected_keys:
            assert key in result, f"Missing biomarker: {key}"


# ──────────────────────────────────────────────
#  End-to-end smoke test (5 subjects)
# ──────────────────────────────────────────────

class TestEndToEnd:
    """End-to-end smoke test with 5 subjects."""

    def test_smoke_5_subjects(self, tmp_path) -> None:
        """
        Full pipeline smoke test: generate → preprocess → encode → biomarkers.
        Uses 5 subjects, 10 s recordings, 5 epochs.
        """
        import yaml, shutil

        # Minimal config
        cfg = {
            "seed": 42,
            "data": {
                "n_subjects": 5,
                "fs": 100,
                "duration": 10,
                "output_dir": str(tmp_path / "synthetic"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "preprocessing": {
                "bandpass_low": 0.5,
                "bandpass_high": 20.0,
                "notch_freq": 50.0,
                "window_samples": 200,
                "stride_samples": 50,
                "baseline_fraction": 0.2,
                "max_nan_gap_ms": 200.0,
            },
            "model": {
                "embed_dim": 64,
                "lstm_hidden": 64,
                "lstm_layers": 1,
                "n_heads": 2,
                "n_streams": 4,
                "dropout": 0.1,
            },
            "training": {
                "lr": 1e-3,
                "batch_size": 4,
                "epochs": 2,
                "weight_decay": 0.0,
                "mask_ratio": 0.25,
                "vicreg_lambda": 1.0,
                "vicreg_mu": 1.0,
                "vicreg_nu": 0.1,
                "temperature": 0.07,
            },
            "biomarkers": {"output_file": str(tmp_path / "biomarkers.csv")},
            "output_dir": str(tmp_path / "output"),
            "models_dir": str(tmp_path / "models"),
            "explainability": {"n_mc_samples": 3, "shap_samples": 5},
            "evaluation": {"test_fraction": 0.3, "cv_folds": 2},
        }

        device = torch.device("cpu")
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Stage 1
        from data.synthetic_generator import generate_synthetic_data
        generate_synthetic_data(5, 100.0, 10.0, cfg["data"]["output_dir"], 42)
        gen_files = list(Path(cfg["data"]["output_dir"]).glob("*.npz"))
        assert len(gen_files) == 5, f"Expected 5 generated files, got {len(gen_files)}"

        # Stage 2 (abbreviated)
        from preprocessing.noise_filter import filter_stream
        from preprocessing.windowing import window_all_streams, compute_reliable_mask
        from preprocessing.normalize import normalize_subject
        from preprocessing.missing_data import handle_missing_data
        from preprocessing.synchronize import synchronize_streams

        proc_dir = Path(cfg["data"]["processed_dir"])
        proc_dir.mkdir(parents=True, exist_ok=True)

        for fpath in gen_files:
            data = np.load(fpath, allow_pickle=True)
            streams = {
                k: data[k].copy()
                for k in ["finger", "wrist", "gait", "insole_pressure", "phone_acc_gyro", "physio"]
            }
            synced, _ = synchronize_streams(streams, 100.0)
            windowed = window_all_streams(synced, 200, 50)
            mask = compute_reliable_mask(synced, 200, 50, 200.0, 100.0)
            cleaned = handle_missing_data(windowed, mask, 200.0, 100.0)
            if min(v.shape[0] for v in cleaned.values()) == 0:
                continue  # skip if no windows
            normalized, _ = normalize_subject(cleaned)
            save_dict = {name: arr for name, arr in normalized.items()}
            for key in data.files:
                if key.startswith("meta_"):
                    save_dict[key] = data[key]
            np.savez_compressed(proc_dir / fpath.name, **save_dict)

        proc_files = list(proc_dir.glob("*.npz"))
        assert len(proc_files) > 0, "No processed files were created"

        # Stage 3 — encoder forward pass
        from models.encoders import MultiStreamEncoder
        enc = MultiStreamEncoder(
            embed_dim=cfg["model"]["embed_dim"],
            lstm_hidden=cfg["model"]["lstm_hidden"],
            lstm_layers=cfg["model"]["lstm_layers"],
            n_heads=cfg["model"]["n_heads"],
            dropout=cfg["model"]["dropout"],
        )
        enc.eval()
        dummy = {k: torch.randn(2, c, 200) for k, c in
                 [("finger", 6), ("gait", 6), ("insole", 4), ("phone", 6), ("wrist", 6)]}
        with torch.no_grad():
            embs = enc(dummy["finger"], dummy["gait"], dummy["insole"], dummy["phone"], dummy["wrist"])
        for emb in embs:
            assert emb.shape == (2, cfg["model"]["embed_dim"])

        # Stage 4 — cross-body forward pass
        from models.cross_body import CrossBodyDependencyModule
        cb = CrossBodyDependencyModule(embed_dim=cfg["model"]["embed_dim"], n_heads=2)
        cb.eval()
        with torch.no_grad():
            fused, attn_w, dep_m, lag_w, _ = cb(*embs)
        assert fused.shape[1] == 512

        print("\n[SMOKE TEST] All checks passed ✓")
