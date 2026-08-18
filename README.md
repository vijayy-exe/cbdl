# CBDL Pipeline

## Cross-Body Dependency Learning for Disease-Agnostic Motor Symptom Monitoring

A complete PyTorch implementation of the 9-stage CBDL framework using synthetic sensor data.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (~20–40 min on CPU with default 200 subjects, 30 epochs)
python main.py

# Fast mode for testing (20 subjects, 5 epochs, ~5 min)
python main.py --fast

# Run tests
pytest tests/ -v
```

---

## Project Structure

```
cbdl_pipeline/
├── config.yaml                        # All hyperparameters
├── main.py                            # Pipeline orchestrator
├── requirements.txt
│
├── data/
│   └── synthetic_generator.py        # Stage 1: Synthetic sensor data
│
├── preprocessing/
│   ├── noise_filter.py               # Stage 2a: Butterworth + notch filter
│   ├── synchronize.py                # Stage 2b: Cross-correlation alignment
│   ├── windowing.py                  # Stage 2c: Sliding window (2s/0.5s stride)
│   ├── normalize.py                  # Stage 2d: Z-score normalization
│   └── missing_data.py               # Stage 2e: NaN interpolation + masking
│
├── models/
│   ├── encoders.py                   # Stage 3: CNN+BiLSTM stream encoders (128-d)
│   ├── cross_body.py                 # Stage 4: CBDL core (5 dependency modules)
│   ├── representation.py             # Stage 5: Self-supervised pretraining
│   └── personalization.py            # Stage 7: FiLM conditioning
│
├── biomarkers/
│   └── compute.py                    # Stage 6: 6 novel digital biomarkers
│
├── explainability/
│   ├── shap.py                       # Stage 8a: KernelSHAP
│   ├── attention_viz.py              # Stage 8b: Attention heatmap
│   └── uncertainty.py               # Stage 8c: MC Dropout uncertainty
│
├── evaluation/
│   └── eval.py                       # Stage 9: Clinical evaluation suite
│
├── training/
│   └── dataset.py                    # PyTorch Dataset + DataLoader
│
└── tests/
    └── test_pipeline.py              # Pytest test suite
```

---

## The 9 Stages

### Stage 1 — Data Sources (Synthetic Generation)
Generates 6 sensor streams for 200 virtual subjects (100 healthy + 100 symptomatic) at 100 Hz × 60 s. Injects realistic noise (Gaussian, 2–5% NaN dropout, timestamp jitter).

| Stream | Channels | Signal Profile |
|--------|----------|---------------|
| finger | 6 | Tremor 3–7 Hz (symptomatic), smooth (healthy) |
| wrist | 6 | Bradykinesia ramps + dyskinesia bursts |
| gait | 6 | Stride cycles with shuffled cadence |
| insole_pressure | 4 | Left/right weight asymmetry |
| phone_acc_gyro | 6 | Trunk sway, postural drift |
| physio | 4 | HR, HRV, skin temp, respiration |

### Stage 2 — Signal Preprocessing
- **Noise filtering**: Butterworth bandpass (0.5–20 Hz) + IIR notch @ 50 Hz
- **Synchronization**: Cross-correlation alignment of all streams to gait
- **Windowing**: 2s windows, 0.5s stride (200/50 samples)
- **Normalization**: Z-score per channel using first 20% windows as baseline
- **Missing data**: Linear interpolation of NaN gaps ≤ 200 ms; longer gaps excluded

### Stage 3 — Multi-Sensor Feature Learning
Four stream encoders, each (B, C, T) → (B, 128):
- **FingerTemporalEncoder**: Conv1D (3 blocks, k=5, 32→64→128) + 2-layer BiLSTM + mean-pool
- **GaitTemporalEncoder**: Same architecture as Finger
- **BalanceEncoder**: Pressure + phone streams concatenated → CNN+BiLSTM
- **SensorEncoder**: Wrist + phone via cross-attention pooling

### Stage 4 — Cross-Body Dependency Learning (Core Novelty)
Five dependency modules operating on 4 stream embeddings [finger, gait, balance, sensor]:

1. **Cross-Limb Attention**: MHA (4 heads) over 4 tokens → attended embeddings + 4×4 weight matrix
2. **Temporal Dependency Graph (TDG)**: Custom pure-PyTorch 2-layer GAT (4 heads) over body-part graph
3. **Dynamic Dependency Matrix (DDM)**: GRU head → 4×4 coupling matrix per window
4. **Time-Lag Correlation**: Learnable shift operator estimates optimal lag (0–250 ms)
5. **Contrastive Dependency Learning**: InfoNCE loss (τ=0.07) between finger–gait pairs

Output: concatenated 512-d cross-body embedding

### Stage 5 — Disease-Agnostic Neuromotor Representation
Self-supervised pretraining (**no disease labels used**):
- **Masked reconstruction**: 25% timestep masking → Transformer decoder reconstruction
- **VICReg loss**: Variance + Invariance + Covariance on 512-d embeddings

### Stage 6 — Novel Digital Biomarkers

| Biomarker | Description |
|-----------|-------------|
| **CBDi** (Cross-Body Dependency Index) | Mean off-diagonal |energy| of 4×4 dependency matrix |
| **Finger–Gait Coupling Score** | Cosine similarity weighted by learned lag |
| **Neuromotor Stability Score** | 1 / variance of cross-body embedding across windows |
| **Coordination Index** | Entropy of attention map distribution |
| **Motor Variability Index** | Mean std of stream embeddings across time |
| **Synchronization Score** | Max cross-correlation peak at learned lag |

### Stage 7 — Personalization
FiLM conditioning: metadata [age_norm, gender, dominant_hand, baseline_norm] → (γ, β) scale and shift applied to 512-d embedding.

### Stage 8 — Explainability & Uncertainty
- **SHAP**: KernelSHAP on Ridge regressor; 6-feature importance summary plot
- **Attention Map**: 4×4 heatmap of cross-limb attention weights
- **Uncertainty**: MC Dropout simulation → mean ± std per biomarker

### Stage 9 — Clinical Evaluation (NOT training)
| Task | Method | Output |
|------|--------|--------|
| UPDRS prediction | Ridge regression (5-fold CV) | RMSE, MAE, R² |
| Fall Risk classification | Logistic regression (5-fold CV) | AUC, accuracy |
| Motor Severity correlation | Spearman correlation of CBDi | r, p-value |

---

## Output Files

After `python main.py`:

| File | Description |
|------|-------------|
| `data/synthetic/*.npz` | 200 subject recordings |
| `models/encoder.pt` | Stream encoder checkpoint |
| `models/crossbody.pt` | Cross-body module checkpoint |
| `models/repr.pt` | Representation module checkpoint |
| `models/personalized.pt` | FiLM personalization checkpoint |
| `output/biomarkers.csv` | 200 rows × 6 biomarkers + metadata |
| `output/shap_summary.png` | SHAP feature importance plot |
| `output/attention_map.png` | Cross-limb attention heatmap |
| `output/uncertainty.csv` | Biomarker uncertainty estimates |
| `output/updrs_metrics.json` | UPDRS RMSE, MAE, R² |
| `output/fallrisk_metrics.json` | Fall Risk AUC, accuracy |
| `output/updrs_scatter.png` | Predicted vs true UPDRS scatter plot |
| `output/fallrisk_roc.png` | Fall Risk ROC curve |
| `output/correlation.csv` | CBDi–Motor Severity Spearman r |
| `output/training_curves.png` | Reconstruction + VICReg loss curves |

---

## Configuration (`config.yaml`)

Key parameters:

```yaml
data:
  n_subjects: 200      # Number of virtual subjects
  fs: 100              # Sampling frequency (Hz)
  duration: 60         # Recording duration (s)

training:
  lr: 0.0003           # Adam learning rate
  batch_size: 64
  epochs: 30
  mask_ratio: 0.25     # Fraction of timesteps masked
  temperature: 0.07    # InfoNCE temperature

model:
  embed_dim: 128       # Per-stream embedding dimension
  cross_body_dim: 512  # Fused cross-body dimension
```

---

## Design Decisions

- **No PyTorch Geometric dependency**: Custom pure-PyTorch GAT eliminates complex install requirements
- **CPU-compatible**: Runs without CUDA
- **Self-contained**: All synthetic data generated internally; no external datasets needed
- **Disease labels isolated**: Labels never used during training, only during Stage 9 evaluation
