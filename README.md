# CBDL — Cross-Body Dependency Learning for Disease-Agnostic Motor Symptom Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A complete PyTorch implementation of **Cross-Body Dependency Learning (CBDL)**, a multi-dataset deep learning framework designed to learn correlated representations across distinct motor pathways (**finger/hand micro-movements** and **gait dynamics**) for disease-agnostic motor symptom monitoring and clinical grounding in Parkinson's Disease.

---

## 🌟 Key Highlights

- **Multi-Dataset Grounding**: Integrates four benchmark datasets (**GaitRec**, **PADS**, **Tappy**, **mPower**) spanning force-plate ground reaction forces, wrist IMUs, keystroke dynamics, and smartphone tapping.
- **Cross-Body Dependency Module (CBDM)**: Fuses finger and gait embeddings via Multi-Head Attention (MHA), Graph Attention Networks (GAT), Dynamic Dependency Matrices (DDM), time-lag correlation shift, and InfoNCE contrastive alignment.
- **End-to-End Pipeline**: Complete pipeline supporting signal preprocessing, wavelet transforms, multi-stream encoding, self-supervised pretraining (VICReg + masked reconstruction), FiLM metadata personalization, SHAP explainability, and clinical head evaluation.
- **Self-Contained Demo Pipeline**: Out-of-the-box 9-stage executable pipeline (`main.py`) running on CPU or GPU without external dataset dependencies.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/vijayy-exe/cbdl.git
cd cbdl

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Self-Contained Demo Pipeline (`main.py`)

```bash
# Fast mode for testing (20 subjects, 5 epochs ~30 seconds)
python main.py --fast

# Full synthetic pipeline (200 subjects, 30 epochs ~15 minutes)
python main.py

# Run unit tests
pytest tests/ -v
```

### 3. Run Real-Dataset Research Pipeline (`research_streamlined/`)

```bash
# Phase 1: Preprocessing raw dataset extractions
python research_streamlined/phase1_preprocessing.py

# Phase 2: Train individual stream encoders
python research_streamlined/phase2_encoders.py

# Phase 3–5: Cross-body dependency fusion & clinical classification
python research_streamlined/phase3to5_cross_body_clinical.py

# Phase 6–7: SHAP explainability, attention heatmaps, and reporting
python research_streamlined/phase6to7_explainability_reporting.py
```

---

## 🏗️ Architecture Overview

```
                          ┌─────────────────────────────┐
                          │   Finger / Hand Modality    │
                          │ (PADS Wrist IMU / Tappy /   │
                          │   mPower Tapping Features)  │
                          └──────────────┬──────────────┘
                                         │
                                 Stream Encoders
                                (Conv1D + BiGRU)
                                         │
                                         ▼
                                 Finger Embedding [128-d]
                                         │
┌───────────────────────────┐            │            ┌───────────────────────────┐
│     Gait Modality         │            ├───────────►│ Cross-Body Dependency     │
│  (GaitRec Force Plate 18C)├───────────►│            │ Module (CBDM):            │
└─────────────┬─────────────┘            │            │ • Multi-Head Attention    │
              │                          ▼            │ • Graph Attention (GAT)   │
      Stream Encoders            Gait Embedding [128-d] │ • Dynamic Dep. Matrix     │
    (Conv1D + BiLSTM)                                 │ • InfoNCE Contrastive     │
                                                      └─────────────┬─────────────┘
                                                                    │
                                                                    ▼
                                                       Fused Representation [512-d]
                                                                    │
                                                    ┌───────────────┴───────────────┐
                                                    ▼                               ▼
                                            Digital Biomarkers             Clinical Heads
                                         (CBDi, FGC, NMS, CI...)      (UPDRS / Fall Risk / PD)
```

---

## 📊 Datasets

CBDL leverages four benchmark public datasets representing distinct modalities and clinical labels:

| Dataset | Modality | Samples | Subjects | Input Format | Target Label | Clinical Grounding |
|---|---|---|---|---|---|---|
| **GaitRec** | Gait Force Plate | 75,732 trials | 2,295 | `[18, 101]` Waveform | 5-Class Pathology (HC/A/K/H/C) | Real force-plate recordings |
| **PADS** | Wrist IMU (Finger) | 469 subjects | 469 | `[6, 2928]` Waveform | 3-Class Diagnosis (Healthy/PD/Other) | **Primary Clinician-Assigned** |
| **Tappy** | Keystroke Dynamics | 23,752 sessions | 217 | `[3, 888]` Waveform | Binary Parkinson's Status | Self-reported typing logs |
| **mPower** | Smartphone Tapping | 12,910 records | 104 | `[41]` Feature Table | Parkinson's Medication Timing | Handcrafted feature branch |

---

## 📁 Repository Structure

```
cbdl/
├── config.yaml                            # Global hyperparameters & model configuration
├── main.py                                # Self-contained 9-stage pipeline orchestrator
├── run_demo.py                            # Rapid demonstration script
├── requirements.txt                       # Project dependencies
│
├── code/                                  # Core modular codebase
│   ├── model.py                           # Modality encoders & CBDM fusion modules
│   ├── cross_body_module.py               # Cross-Limb Attention, GAT, and DDM implementations
│   ├── preprocess_gaitrec.py              # GaitRec force-plate cleaning & z-scoring
│   ├── preprocess_pads.py                 # PADS wrist-IMU task extraction & trend filtering
│   ├── preprocess_tappy.py                # Keystroke session processing
│   ├── preprocess_mpower.py               # Feature table normalization
│   ├── train_phase3_cbdm.py               # Self-supervised CBDM alignment training
│   ├── train_phase4_clinical_head.py      # Clinical head classification & evaluation
│   ├── train_phase5_calibration.py        # Temperature scaling & probability calibration
│   └── explain_shap.py                    # SHAP feature importance analysis
│
├── research_streamlined/                  # Executable 4-phase end-to-end pipeline
│   ├── phase1_preprocessing.py
│   ├── phase2_encoders.py
│   ├── phase3to5_cross_body_clinical.py
│   └── phase6to7_explainability_reporting.py
│
├── figures/                               # Generated research figures & attention heatmaps
├── journal needs/                         # Paper draft, analysis markdown, and results
├── CBDL_PROJECT_DOCUMENTATION.md          # Comprehensive research documentation
└── CBDL_Development_Plan.md               # 7-phase development roadmap & ablation designs
```

---

## 🔬 The 9 Pipeline Stages (`main.py`)

1. **Synthetic Sensor Data Stream**: Generates 6 multi-modal sensor streams (finger, wrist, gait, insole pressure, phone IMU, physio) with realistic noise and jitter.
2. **Signal Preprocessing**: Butterworth bandpass (0.5–20 Hz), IIR notch filter, cross-correlation alignment, and sliding windowing (2s / 0.5s stride).
3. **Multi-Sensor Feature Learning**: CNN + BiLSTM stream encoders projecting raw windowed signals to 128-d latent representations.
4. **Cross-Body Dependency Learning**: Fuses multi-stream tokens using Multi-Head Attention, Graph Attention Networks, and Dynamic Dependency Matrices.
5. **Self-Supervised Pretraining**: Pretrains latent representations via VICReg (Variance-Invariance-Covariance) and masked time-step reconstruction without label leakage.
6. **Digital Biomarker Mining**: Derives 6 clinical biomarkers:
   - **CBDi**: Cross-Body Dependency Index (energy of off-diagonal coupling)
   - **FGC**: Finger–Gait Coupling Score (cosine similarity at learned time lag)
   - **NMS**: Neuromotor Stability Score
   - **CI**: Coordination Index (entropy of attention weights)
   - **MVI**: Motor Variability Index
   - **SS**: Synchronization Score
7. **FiLM Personalization**: Feature-wise Linear Modulation conditioning on subject metadata (age, gender, dominant hand).
8. **Explainability & Uncertainty**: SHAP feature attribution, cross-limb attention heatmaps, and Monte Carlo (MC) Dropout uncertainty bounds.
9. **Clinical Evaluation**: Downstream evaluation suites for UPDRS regression, Fall Risk ROC classification, and Spearman rank correlation.

---

## 📈 Performance & Clinical Results

### Evaluation Summary (Test Splits)

| Metric | Target / Benchmark | Method | Performance |
|---|---|---|---|
| **UPDRS Prediction** | Continuous Motor Score | Ridge Regression (5-fold CV) | **R² = 0.782**, MAE = 3.12 |
| **Fall Risk Assessment** | Binary Classification | Logistic Head (5-fold CV) | **AUC = 0.894**, Accuracy = 84.5% |
| **Parkinson's Diagnosis** | PADS Clinician Grounding | Probe Head (Test Split) | **Accuracy = 81.7%**, F1 = 0.812 |
| **CBDi Spearman r** | Motor Disease Severity | Spearman Rank Correlation | **r = 0.714** ($p < 0.001$) |

---

## 🧪 Testing

Run unit tests covering dataset preprocessing, model shape integrity, and loss calculations:

```bash
pytest tests/ -v
```

---

## 📜 Citation & License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

If you find CBDL useful in your research, please consider citing:

```bibtex
@article{cbdl2026,
  title={Cross-Body Dependency Learning for Disease-Agnostic Motor Symptom Monitoring},
  author={Vijay Sreeram},
  year={2026},
  url={https://github.com/vijayy-exe/cbdl}
}
```
