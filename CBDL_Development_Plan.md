# CBDL — Phase-Wise Implementation Plan (Updated)
### Cross-Body Dependency Learning for Disease-Agnostic Motor Symptom Monitoring
**Scoped to: GaitRec + Tappy + PADS + mPower confirmed | PPMI / Human3.6M / NTU RGB+D dropped | ~18-day runway to Sep 18 submission**

---

## 0. Reality Check Before You Build Anything

The architecture in your original PDF is a *publication-grade, multi-year* system: self-supervised pretraining on two large pose datasets, a novel lagged cross-attention + Soft-DTW + contrastive module, clinical grounding on PPMI, FiLM personalization, calibration, and SHAP explainability. Built fully, that is a strong paper. Built fully **in 18 days with 4 confirmed datasets**, it is not — and trying to force every original stage to completion will most likely leave you with a half-working pipeline and no results section.

The good news: your dataset lineup (GaitRec, Tappy, PADS, mPower) is a genuinely solid basis for a scoped methods paper. PADS carries real clinically-assigned diagnostic labels (PD / healthy control / differential diagnosis), replacing PPMI's supervision role with something better than a proxy label. mPower adds a fourth finger-movement signal — but it arrived as pre-computed tapping features (`tapFeatures.tsv`), not raw waveform data, so it's incorporated differently from Tappy/PADS: as a parallel handcrafted-feature branch fused into the Finger representation, not forced into the shared waveform trunk. See Phase 2 for exactly how.

**Two tracks:**
- **Track A — Core Deliverable (mandatory, ~12 days):** working, evaluated, honestly-scoped CBDL running end-to-end on GaitRec + Tappy + PADS + mPower.
- **Track B — Stretch Additions (only if time remains, ~last 4–6 days):** calibration, SHAP, FiLM personalization — attempted *after* Track A produces results, not before.

Do not start Stage 3 (the novelty module) until the encoders run cleanly on real batches. A broken pipeline with a fancy attention module is worse than a simple pipeline that runs.

---

## Final Architecture (Locked)

```
Tappy (keystroke)  ─┐
                     ├─► [per-source adapter] ─► Finger Waveform Encoder ─┐
PADS (wrist IMU)   ──┘                              (shared TCN/BiGRU)    │
                                                                            ├─► [fusion layer] ─► Finger Embedding ─┐
mPower (tapFeatures.tsv,                                                   │                                        │
   handcrafted features) ──────► [small MLP branch] ───────────────────────┘                                        │
                                                                                                                       ├─► Cross-Body Dependency Module
GaitRec (gait)     ─────────────────────────► Gait Encoder (CNN-LSTM/Transformer) ───────────────────────────────────┘         (Lagged Cross-Attention +
                                                                                                                                   Soft-DTW + Contrastive
                                                                                                                                   Lag Learning)
                                                                                                                                         │
                                                                                                 ┌───────────────────────────────────────┴──────────────────────┐
                                                                                       Clinical Grounding                                              (Track B, if time allows)
                                                                               (PADS diagnostic labels: PD/HC/DD)                     FiLM personalization → Calibration → SHAP
                                                                                                 │
                                                                                       Classifier / Risk output
```

No separate Body Pretraining Encoder — dropped, see Phase 0.3. mPower enters through its own small MLP branch (not the shared waveform trunk) because it only provided pre-computed tapping features, not raw signal — see Phase 0.1 and Phase 2 for why this design, not a compromise.

---

## Phase 0 (Day 0–1): Dataset Finalization & Substitution Strategy

### 0.1 Finger modality — Tappy + PADS as waveform sources, mPower as a handcrafted-feature branch

Tappy (keystroke dynamics) and PADS (wrist-IMU finger-tapping task) are pooled as two acquisition views of the same construct — finger micro-movement — through the shared **Finger Waveform Encoder**, each entering via its own small **input adapter** (a linear/1D-conv projection into a common embedding width) before the shared TCN/BiGRU trunk.

mPower is real, confirmed data now, but it arrived through the Public Researcher Portal as `tapFeatures.tsv` — **pre-computed tapping features** (summary statistics per session), not raw touch/accelerometer signal. Two honest paths existed here:
- Query mPower's raw per-tap Synapse **table** (not the file bundle) to get true waveform data matching Tappy/PADS's format — the "purist" option, but adds a separate access/query step you don't have guaranteed time for.
- **(Chosen) Keep `tapFeatures.tsv` as-is and give it its own small MLP branch**, fused with the waveform encoder's output via a fusion layer before producing the final Finger Embedding. This is standard practice in multimodal ML (combining a deep sequence encoder with a handcrafted-feature branch) — it's not a workaround, it's the honest architecture for what mPower actually gave you.

State this plainly in the paper: *"the Finger representation combines a shared sequence encoder over Tappy and PADS waveforms with a parallel handcrafted-feature branch over mPower's pre-extracted tapping features, fused via a linear projection layer."* That's accurate and unremarkable to a reviewer — heterogeneous multi-source fusion is common, hiding that one source is feature-level rather than raw would not be.

**mPower's diagnosis label:** `tapFeatures.tsv` alone may not carry a PD/non-PD label — check its columns first (`professional-diagnosis` or similar). If absent, pull mPower's separate demographics/enrollment module and join on `healthCode`. Either way, mPower's label is a secondary signal — PADS remains the primary clinical grounding source (0.2 below), since its labels are clinician-assigned rather than self-reported.

### 0.2 Clinical grounding — PADS diagnostic labels replace PPMI

PPMI access isn't confirmed and isn't worth waiting on. PADS carries genuine clinically-assigned diagnostic group labels (PD / HC / DD) from its own study — that becomes your primary clinical grounding signal (Stage 4), applied to the final Finger Embedding (waveform trunk + mPower fusion combined), since PADS is a finger/hand-task dataset, not gait. GaitRec's own pathology / patient-vs-healthy metadata serves as a secondary gait-side check, since PADS doesn't cover lower-limb gait.

State this in the paper plainly: *"clinical grounding is supervised via PADS's diagnostic group labels (PD/HC/DD), used in place of PPMI's MDS-UPDRS severity regression due to data-access constraints."* Real classification labels, not a proxy — a defensible, credible substitution.

### 0.3 Full-body pretraining (Human3.6M / NTU RGB+D) — dropped, not needed

None of GaitRec, Tappy, or PADS contain full-body pose sequences — GaitRec is gait kinematics/kinetics, Tappy is keystroke timing, PADS is wrist-IMU. There is no pose-sequence input anywhere in the pipeline for a Body Pretraining Encoder to pretrain on, and multi-day self-supervised pretraining isn't viable in this window regardless. The architecture is finalized as **two encoders — Finger, Gait —** feeding directly into the Cross-Body Dependency Module. Human3.6M/NTU stay out unless a genuine pose-sequence data source is added later.

### Deliverable for Phase 0
A one-paragraph "Dataset and Scope" note for the paper: datasets used (GaitRec, Tappy, PADS, mPower), how each is incorporated (Tappy+PADS via a shared waveform encoder with per-source adapters; mPower via a parallel handcrafted-feature MLP branch, since only pre-extracted features were available; PPMI→PADS diagnostic labels; Human3.6M/NTU→dropped, no pose modality present), and what's untouched (the Cross-Body Dependency Module itself — your core novelty).

---

## Phase 1 (Day 1–3): Data Acquisition & Preprocessing — 1D Waveform Representation (+ mPower Tabular Cleaning)

**Goal:** three clean, subject-indexed **1D signal tensors** (GaitRec, Tappy, PADS) plus one clean **tabular feature table** (mPower). Not everything collapsed into summary statistics — only mPower is tabular, because that's the form it actually arrived in.

**Why waveforms, not scalars, for the other three.** The encoders (TCN/BiGRU for finger, CNN-LSTM/Transformer for gait) are sequence models. Feeding them pre-computed scalars per session (mean tap interval, mean cadence) collapses temporal structure before the model sees it, defeats the point of using sequence models at all, and gives the Cross-Body Dependency Module's lagged attention and Soft-DTW nothing to actually align. Treat GaitRec, Tappy, and PADS like audio signals: a 1D array ordered in time, with a few parallel channels, not a bag of statistics. mPower is the one exception, handled separately in step 5 below, precisely because raw waveform data wasn't what was available.

1. **Inspect all four datasets first**, before writing preprocessing code. Check sample counts, per-subject recording counts, sampling rates, label formats, missingness, and native time resolution of each. Write these numbers down for the paper's Dataset section.

2. **GaitRec — multichannel 1D waveform:**
   - Channels as continuous 1D arrays over the gait cycle/trial (e.g. vertical ground-reaction-force trace, step-timing/cadence trace, stride-length trace) — use the sensor-level series GaitRec provides directly, not derived per-trial scalars.
   - Resample every trial to a common length via interpolation (100–200 timesteps per gait cycle is standard in gait-analysis literature).
   - Result shape: `[channels, timesteps]`, e.g. `[3, 150]`.

3. **Tappy — 1D waveform from the keystroke event stream:**
   - Convert event-based logs (timestamp + key + hold-time) into parallel 1D channels: inter-key-interval sequence, hold-time sequence, flight-time/latency sequence — ordered arrays across a session, not session-averaged scalars.
   - Pad/truncate to a common sequence length.
   - Result shape: `[channels, timesteps]`, same convention as the other two waveform sources.

4. **PADS — multichannel wrist-IMU waveform:**
   - Isolate the finger/hand-relevant task windows per subject (PointFinger, TouchIndex, TouchNose — not the other ~8 PADS tasks, which are gait/postural/whole-arm movements).
   - 6-axis IMU (accel + gyro) per wrist; use one wrist (simpler, defensible) or both (up to 12 channels) — pick one and state it.
   - Resample to the same fixed-length convention used above.
   - Keep the diagnostic label (PD/HC/DD) attached per subject for Phase 4.

5. **mPower — clean as a tabular feature table, not a waveform:**
   - Load `tapFeatures.tsv`, inspect columns, and check whether a diagnosis label (e.g. `professional-diagnosis`) is already present. If not, pull mPower's demographics/enrollment module and join on `healthCode`.
   - Standard tabular cleaning: drop rows with excessive missingness, impute or drop remaining nulls (document which), remove obvious outliers/invalid sessions (e.g. zero or implausible tap counts).
   - Normalize each feature column independently (z-score, fit on train split only).
   - This stays a flat `[N_subjects, num_features]` table — do not attempt to reshape it into a `[channels, timesteps]` waveform; there's no genuine time axis in pre-aggregated features, and faking one would misrepresent the data.

6. **Subject alignment:** GaitRec, Tappy, PADS, and mPower are four disjoint subject pools — no shared IDs across any pair. True within-subject cross-body modeling isn't possible across all four. Use **population-level cross-body learning**: the Cross-Body Dependency Module learns finger→gait temporal relationships across the population, grouped by diagnosis/severity tier (grounded in PADS's real labels, not a pure proxy) rather than literal per-subject pairing. State this explicitly in Limitations.

7. Hold out a **subject-disjoint** validation/test split per dataset — never split by sample.

**Deliverable:** three `.npz`/`.pt` waveform tensor stores (`[N_samples, channels, timesteps]` for GaitRec, Tappy, PADS) plus one cleaned mPower feature table (`.csv`/`.npy`), a data card per dataset (channel/feature definitions, resampling or cleaning decisions), and a train/val/test subject split file per dataset. See the storage-format note below — this is the format your training loop will actually read from.

### Storage format for training (speed)
Cache preprocessed waveform tensors as `.npy` or `torch.save` (`.pt`) and the mPower feature table as `.npy` or `.csv` — **not** raw CSV/JSON re-parsing inside the training loop for any of them. Use float32, fixed-length sequences for the waveform sources (no runtime padding), load everything into RAM once at training start via a plain `TensorDataset` (or a small custom `Dataset` combining waveform + tabular tensors per subject), `num_workers=0–2`, `pin_memory=True` on GPU. Preprocess once — at your data scale this is the single biggest time-saver available.

---

## Phase 2 (Day 3–5): Modality Encoders

1. **Per-source adapters:** a small linear/1D-conv layer per waveform-based finger source (Tappy adapter, PADS adapter) that projects each source's raw channel count into a common embedding width before entering the shared trunk.

2. **Finger Waveform Encoder:** TCN or BiGRU, shared trunk fed by both adapters (Tappy + PADS pooled). Output a fixed-size embedding (64–128 dim).
   - **Sanity-check against PADS's real diagnostic label** via a small linear probe — genuine validation, since PADS's labels are clinically assigned. If the probe can't beat a majority-class baseline, stop and debug before building anything on top.

3. **mPower feature branch (separate from the waveform trunk):** a small MLP (2–3 layers) over `tapFeatures.tsv`'s handcrafted features, output to the same embedding width as the waveform encoder. Keep this branch simple — it's a handful of tabular features, not a sequence, so an oversized MLP just overfits.

4. **Fusion layer:** concatenate the Finger Waveform Encoder's output with the mPower MLP branch's output, then a single linear layer projects the concatenation down to the final Finger Embedding dimension. This is the one place the two finger "views" (sequence-based and feature-based) actually meet.

5. **Gait Encoder:** CNN-LSTM or small Transformer over the GaitRec sequence, same output dimensionality as the final Finger Embedding (needed for Stage 3's cross-attention). Linear-probe sanity check against GaitRec's pathology label.

6. Keep every component small (few hundred thousand parameters total) — limited data, limited time; an oversized model overfits and eats days in debugging.

**Deliverable:** a trained Finger pathway (waveform trunk + mPower MLP branch + fusion layer) and a trained Gait Encoder, with validation-set probe accuracy reported for each. The Finger pathway's probe result against PADS's real labels is worth a paper figure — it's your strongest evidence the learned representation is clinically meaningful. As a secondary check, also report the probe accuracy of the waveform-only pathway (before mPower fusion) vs. the fused pathway — this is a free, easy ablation showing whether adding the mPower feature branch actually helps, which strengthens the paper's evaluation section.

---

## Phase 3 (Day 5–9): Cross-Body Dependency Module — core novelty, keep this intact

This is the stage reviewers/examiners will scrutinize hardest — don't cut corners here even though everything upstream was simplified.

1. **Lagged Cross-Attention:** attention between finger embeddings at time *t* and gait embeddings at time *t+Δ* for a small set of candidate lags (Δ ∈ {0, 1, 2, 3 windows}). Population-level pairing means you're learning *which lag best explains cross-body relationships across the cohort*, not a literal per-patient delay — state that framing explicitly.
2. **Soft-DTW alignment:** use an existing differentiable Soft-DTW implementation (don't write one from scratch) to align sequences of differing effective length/tempo.
3. **Contrastive Lag Learning:** positive pairs (matched finger/gait sequences from the same diagnosis/severity tier) vs. negative pairs (randomly mismatched tier), trained with an InfoNCE-style contrastive loss — so the module learns genuine tier-consistent temporal structure, not noise.
4. Output: a single fused embedding per (finger-window, gait-window) pair.

**Deliverable:** trained fusion module, an attention-lag heatmap (which lag gets the most weight across the cohort — a strong paper figure), and an ablation placeholder (filled in Phase 6) comparing fused vs. single-modality performance.

---

## Phase 4 (Day 9–10): Clinical Grounding (PADS-supervised)

- Classification head on top of the fused embedding, supervised primarily by **PADS's diagnostic labels (PD/HC/DD)** — real clinical supervision, the main authenticity strength of this plan. GaitRec's pathology metadata as a secondary gait-side check.
- **FiLM personalization:** Track B — only attempt if Phases 1–3 finished on schedule. If behind by Day 10, skip and note it as "planned but deferred due to time constraints," not silently dropped.

**Deliverable:** working classifier head, held-out test performance (accuracy/F1), and a note on whether FiLM was implemented.

---

## Phase 5 (Day 10–11, Track B): Calibration & Uncertainty

- Isotonic Regression or Platt Scaling on validation-set predicted probabilities.
- One reliability-diagram figure (predicted probability vs. observed frequency) — cheap if time allows, reviewers like calibrated medical-AI claims.
- Skip bootstrap confidence intervals unless there's spare time in the last two days.

---

## Phase 6 (Day 11–13, Track B): Explainability

- **SHAP attribution** on the classifier head's inputs — usually a half-day task once the pipeline runs cleanly.
- **Attention visualization:** reuse the Phase 3 lag-attention heatmap — don't build a second explainability artifact.
- Skip the full Risk Tier / Trajectory Forecast clinical-output UI from the original PDF — a results table and a couple of figures communicate this for a paper; a polished dashboard burns days you don't have.

---

## Publication Feasibility: 4 Datasets, One With Real Clinical Labels, One Feature-Level

**Yes, publishable — venue tier is the variable you control, not dataset count.**

1. **Never claim per-patient longitudinal cross-body coupling** — no shared subjects across GaitRec, Tappy, PADS, or mPower. Defensible claim: *"we demonstrate a cross-body dependency learning framework using population-level, diagnosis-tier-aligned finger and gait signals, clinically grounded via PADS's expert-assigned diagnostic labels, as a proof-of-concept for the architecture; per-subject longitudinal validation is future work pending a paired dataset."*
2. **State the heterogeneous fusion explicitly** — the Finger representation combines a sequence encoder (Tappy + PADS waveforms) with a handcrafted-feature branch (mPower's `tapFeatures.tsv`), fused via a linear projection. This is normal multimodal practice, but only if disclosed; describing all four sources as if uniformly processed would misrepresent the pipeline.
3. **State every substitution explicitly** in the paper (PPMI→PADS labels, dropped pretraining) — disclosed, justified gaps are normal; hidden ones get penalized.
4. **Don't overclaim clinical readiness** — no "clinically validated" or "diagnostic tool" language. Frame as a methodological/architectural contribution to digital biomarker research, evaluated on real diagnostic classification labels (PADS), not MDS-UPDRS severity regression.
5. **Venue reality for 18 days:** a top-tier journal (Movement Disorders / npj Digital Medicine–tier) is not realistic — those need months of review and validated clinical labels you don't have. Realistic and genuinely achievable: a student/undergraduate research symposium, a national/regional IEEE/ACM conference track, a workshop paper, or an arXiv preprint followed by conference submission. These venues welcome scoped, honestly-limited proof-of-concept architecture papers with a clear novel module and a proper ablation study.

---

## Phase 7 (Day 13–17): Evaluation, Ablation, and Paper Writing

Most of your remaining time should go here — a working model with a thin evaluation section is a weaker paper than a simpler model with rigorous evaluation.

1. **Baselines:** finger-only (no gait, no fusion), gait-only (no finger, no fusion), naive concatenation fusion (no lagged attention, no Soft-DTW, no contrastive learning), and finger-waveform-only vs. finger-waveform-plus-mPower (isolates whether the mPower feature branch actually helps) — together these isolate how much the Cross-Body Dependency Module and the mPower fusion specifically contribute.
2. **Ablation table:** full CBDL vs. (no lagged attention) vs. (no Soft-DTW) vs. (no contrastive loss).
3. **Write Limitations honestly and early:** mPower absence (with the adapter-based extensibility noted as a concrete follow-up, not vague future work), PADS-based clinical grounding in place of PPMI, dropped pretraining stage, population-level (not per-subject) cross-body pairing.
4. **Redraw the architecture diagram** to match what was actually built (dotted-out Body Pretraining Encoder box, footnote on mPower/PADS/Tappy roles) — a diagram that doesn't match the build undermines credibility.

---

## Quick Reference: What Changed From the Original Design

| Original stage | Status in this plan | Why |
|---|---|---|
| mPower (finger data) | In hand as `tapFeatures.tsv`; integrated via a separate MLP branch, not the shared waveform trunk | Only pre-computed features were available, not raw signal — feature-fusion is the honest architecture for that |
| PPMI (clinical labels) | Replaced with PADS's clinically-assigned diagnostic labels (PD/HC/DD) | Real labels, not proxy — access to PPMI not confirmed |
| Human3.6M / NTU RGB+D pretraining | Dropped entirely | No pose modality in any confirmed dataset; multi-day compute cost |
| Lagged Cross-Attention, Soft-DTW, Contrastive Lag Learning | **Kept as-is — this is the core novelty** | Untouched by all substitutions above |
| FiLM personalization | Track B, cut first if behind schedule | Refinement, not core contribution |
| Calibration | Track B | Cheap if time allows |
| SHAP + attention visualization | Track B, attention visualization is nearly free (reuse Phase 3 output) | |
| Risk Tier / Trajectory Forecast UI | Dropped from build scope, kept as results table + figures | Not what a paper needs |

If a reviewer or your guide pushes back on any substitution: *stated dataset-access constraint → documented substitute → explicit limitation → named, concrete follow-up.* That is a completely normal and defensible pattern in applied ML papers.
