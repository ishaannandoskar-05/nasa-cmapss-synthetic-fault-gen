# NASA CMAPSS Synthetic Fault Generation — Implementation Summary

## Project Goal
Generate realistic synthetic fault data for supervised classifier training
using a Conditional GAN on NASA CMAPSS turbofan engine sensor data,
since no explicit fault labels exist in the raw dataset.

---

## Dataset
- **Source:** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
- **Subset used:** FD001 (single fault mode, 1 operating condition)
- **Files:** train_FD001.txt, test_FD001.txt, RUL_FD001.txt
- **Sensors:** 21 raw sensors + 3 operational settings per cycle
- **Engines:** 100 training engines, variable length run-to-failure cycles

---

## Phase 1 — Data Ingestion & EDA
**Notebook:** `01_data_quality_and_rul.ipynb`
**Script:** `src/ingestion/load_data.py`

- Loaded raw CMAPSS txt files using whitespace-separated parser
- Assigned column names: engine_id, cycle, op1-3, s1-s21
- Computed RUL (Remaining Useful Life) per engine by counting backwards from end of life
- Saved `train_with_rul.csv` to `data/processed/FD001/`
- Identified 7 near-constant sensors with low variance:
  s1, s5, s6, s10, s16, s18, s19

---

## Phase 2 — Preprocessing
**Notebook:** `02_preprocessing.ipynb`
**Script:** `src/preprocessing/preprocess.py`

- Dropped 7 flat sensors → 14 remaining features (op1-3 + 11 sensors)
- Applied MinMaxScaler per engine unit (not globally) to prevent leakage
- Capped RUL at 125 cycles (piecewise linear target)
- Applied sliding window: W=30 cycles, stride=1
- Output shape: X_train.npy (N, 30, 17), y_train.npy (N,)

> Note: INPUT_DIM=17 (14 sensors + 3 op conditions). Scaler not saved to disk —
> needs fixing before Phase 8 test evaluation.

---

## Phase 3 — Physics-Based Fault Labeling
**Notebook:** `03_physics_labels.ipynb`

Mapped RUL zones to degradation severity classes based on turbofan physics:

| Class | RUL Range | Degradation State      |
|-------|-----------|------------------------|
| C0    | > 100     | Healthy                |
| C1    | 50–100    | Early wear             |
| C2    | 10–50     | Advanced fault         |
| C3    | < 10      | Imminent failure       |

- Class distribution: C0=43%, C1=28%, C2=23%, C3=6%
- Verified physics monotonicity on real data (T50↑, Nf↓, Ps30↓, phi↓)
- Saved `labels_train.npy` and `class_distribution.csv`

---

## Phase 4 — Model Architecture (TimeGAN)
**Notebook:** `04_model_architecture.ipynb`

Defined Conditional TimeGAN with 5 components:

| Component     | Role                                          |
|---------------|-----------------------------------------------|
| Embedder      | Real sequences → latent space (GRU)           |
| Recovery      | Latent space → reconstructed sequences (GRU)  |
| Supervisor    | Enforces temporal coherence in latent space   |
| Generator     | Noise + class embedding → latent sequences    |
| Discriminator | Classifies real vs synthetic latent sequences |

- Conditioning: class embedding (Embedding layer, embed_dim=8)
- Saved architecture config to `configs/model_config.json`

---

## Phase 5 — Training (TimeGAN + CGAN)
**Notebooks:** `05_training_timegan.ipynb`, `05b_training_cgan.ipynb`

### TimeGAN Training (3-phase)
- Phase 1: Autoencoder pretraining (100 epochs)
- Phase 2: Supervisor pretraining (100 epochs)
- Phase 3: Joint GAN training (300 epochs, grad clipping=1.0)
- HIDDEN_DIM=48, LATENT_DIM=48, NUM_LAYERS=3
- **Result:** Mode collapse — generator output visually flat/collapsed,
  discriminative score=1.0, MMD poor across all classes

### Architecture iterations attempted:
1. Sigmoid → Tanh in Embedder/Supervisor/Generator proj layers
2. noise_scale increased 0.1 → 0.3 in Generator
3. Recovery Sigmoid → Linear + clamp(0,1)
4. Reconstruction loss weight reduced 10.0 → 5.0
5. Moments loss gamma increased 1.0 → 2.0

### CGAN (Direct sensor space — current active model)
- Single training loop (500 epochs)
- Generator: FC(noise+embed) → GRU → FC → Sigmoid
- Discriminator: GRU(sequence+embed) → FC → logit
- Feature matching loss added (weight=10.0) to prevent mode collapse
- LR_G=1e-4, LR_D=2e-4, betas=(0.5, 0.999)
- D trained 1x, G trained 2x per batch
- Checkpoints saved to `data/processed/FD001/checkpoints_cgan/`

---

## Phase 6 — Synthetic Fault Generation
**Notebook:** `06_synthetic_fault_generation.ipynb`

- Generated synthetic windows per class to balance dataset
- Target count = max(real class count) = C0 count (7633)
- C0: 0 generated (already majority), C1/C2/C3: oversampled
- Severity interpolation: C1↔C2 and C2↔C3 midpoints (500 each)
- Combined real + synthetic into balanced dataset
- Saved:
  - `data/synthetic/GAN/synth_X.npy`
  - `data/synthetic/GAN/synth_labels.npy`
  - `data/synthetic/GAN/interp_C1_C2.npy`
  - `data/synthetic/GAN/interp_C2_C3.npy`
  - `data/processed/FD001/X_balanced.npy`
  - `data/processed/FD001/labels_balanced.npy`

---

## Phase 7 — Validation (TimeGAN results)
**Notebook:** `07_validation.ipynb`

| Metric              | Result  | Target        | Verdict     |
|---------------------|---------|---------------|-------------|
| MMD C1              | 0.129   | < 0.05        | POOR        |
| MMD C2              | 0.712   | < 0.05        | POOR        |
| MMD C3              | 0.826   | < 0.05        | POOR        |
| KS pass rate        | 5.9%    | > 50%         | POOR        |
| Discriminative score| 1.000   | ~0.50         | POOR        |
| Monotonicity        | 1/4     | 4/4           | PARTIAL     |

- PCA/t-SNE: real and synthetic form completely separate clusters
- Decision: TimeGAN rejected, CGAN adopted

> Phase 7 to be re-run after CGAN Phase 5b + Phase 6 completion.

---

## Phases Remaining

| Phase | Notebook                  | Status        |
|-------|---------------------------|---------------|
| 5b    | 05b_training_cgan.ipynb   | In progress   |
| 6     | 06_synthetic_...ipynb     | Needs rerun   |
| 7     | 07_validation.ipynb       | Needs rerun   |
| 8     | 08_classifier.ipynb       | Not started   |
| 9     | 09_cross_dataset_eval.ipynb| Not started  |

---

## Tech Stack
- Python 3.13
- PyTorch (GAN training)
- scikit-learn (preprocessing, discriminative score)
- scipy (KS test)
- pandas / numpy (data handling)
- matplotlib (visualization)
- Jupyter Notebooks
