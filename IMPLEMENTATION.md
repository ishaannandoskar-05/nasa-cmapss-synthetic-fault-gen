# NASA CMAPSS Synthetic Fault Generation - Implementation

## Project Goal

This project builds a synthetic-data-assisted fault classification pipeline for
NASA CMAPSS turbofan engine data. Raw CMAPSS files provide degradation
trajectories and RUL labels, but not explicit fault classes. The project maps
RUL into four degradation classes, trains Conditional GANs to balance minority
fault states, and trains a 1D-CNN classifier for fault-state prediction.

The repository now also includes a Flask API and React dashboard for local
inference and visualization.

## Dataset

**Source:** NASA CMAPSS turbofan degradation simulation

| Subset | Fault modes | Operating conditions | Train engines | Test engines |
|--------|-------------|----------------------|---------------|--------------|
| FD001  | 1           | 1                    | 100           | 100          |
| FD002  | 1           | 6                    | 260           | 259          |
| FD003  | 2           | 1                    | 100           | 100          |
| FD004  | 2           | 6                    | 249           | 248          |

Each raw row contains `engine_id`, `cycle`, three operating settings, and 21
sensor readings.

## Active Feature Set

The active model uses 16 normalized features:

```text
op1, op2,
s2, s3, s4, s7, s8, s9,
s11, s12, s13, s14, s15, s17, s20, s21
```

Seven low-variance sensors are dropped: `s1`, `s5`, `s6`, `s10`, `s16`, `s18`,
and `s19`. `op3` is also excluded from the active FD001-style feature pipeline
because it is constant for that subset.

## Fault Labels

RUL is capped at 125 cycles and mapped into four classes:

| Class | RUL range | Meaning |
|-------|-----------|---------|
| C0    | `> 100`   | Healthy |
| C1    | `51-100`  | Early wear |
| C2    | `11-50`   | Advanced fault |
| C3    | `0-10`    | Imminent failure |

The helper implementation lives in `src/phase3_physics_labels/labeler.py`.

## Notebook Pipeline

| Phase | Notebook | Purpose | Status |
|-------|----------|---------|--------|
| 1 | `01_data_quality_and_rul.ipynb` | Load CMAPSS and compute train RUL | Done |
| 2 | `02_preprocessing.ipynb` | Normalize, cap RUL, create windows | Done |
| 3 | `03_physics_labels.ipynb` | Assign degradation classes | Done |
| 4 | `04_model_architecture.ipynb` | Define model architecture | Done |
| 5 | `05_training.ipynb` | TimeGAN experiment | Deprecated |
| 5b | `05b_training_cgan.ipynb` | FD001 CGAN training | Done |
| 6 | `06_synthetic_fault_generation.ipynb` | Generate balanced synthetic FD001 data | Done |
| 7 | `07_validation.ipynb` | Validate synthetic data | Done |
| 8 | `08_classifier.ipynb` | Train FD001-only classifier | Done |
| 9 | `09_cross_dataset_eval.ipynb` | Test FD001 model across subsets | Done |
| 10 | `10_multi_subset_preprocessing.ipynb` | Process FD002-FD004 | Done |
| 11 | `11_multi_subset_cgan_colab.ipynb` | Multi-subset CGAN training | Done |
| 12 | `12_multi_subset_generation.ipynb` | Build unified balanced dataset | Done |
| 13 | `13_unified_classifier_colab.ipynb` | Train unified classifier | Done |

## Phase Summary

### Data Ingestion

`src/ingestion/load_data.py` loads whitespace-separated CMAPSS train/test files,
assigns stable column names, loads RUL files, and prints basic dataset summaries.

### Preprocessing

`src/preprocessing/preprocess.py` caps RUL, applies per-engine scaling, creates
30-cycle sliding windows, and saves NumPy arrays. The more complete notebook
pipeline saves the scaler artifacts and feature configuration used later by
validation, classifier training, and the app.

Window shape:

```text
(num_windows, 30, 16)
```

### Synthetic Data Generation

TimeGAN was attempted first but deprecated because it collapsed on this sensor
sequence task. The active approach is a direct sensor-space Conditional GAN:

- Generator input: random noise plus class embedding
- Generator backbone: GRU sequence generator
- Discriminator backbone: GRU sequence discriminator
- Output: normalized 30-step, 16-feature sensor windows

Separate CGANs were trained for FD001, FD002, FD003, and FD004. FD002 and FD004
use a discriminator-throttled variant to handle six operating conditions.

### Validation

Validation uses MMD, KS tests, PCA/t-SNE plots, monotonicity checks, and a
discriminative score. FD001 CGAN outputs achieved good MMD but weak KS pass rate,
with a known cross-sensor correlation gap.

### Classifier

The classifier is a 1D-CNN over `(time, features)` windows.

Unified classifier architecture:

- Conv1d `16 -> 64`
- Conv1d `64 -> 128`
- Conv1d `128 -> 256`
- Adaptive average pooling
- Fully connected classifier `2048 -> 512 -> 128 -> 4`
- Dropout in dense layers

The trained model artifact is available at:

```text
data/processed/unified/unified_classifier_1dcnn.pt
turbofan_app/backend/model_artifacts/unified_classifier_1dcnn.pt
```

## Reported Results

### FD001-only Classifier

| Metric | Value |
|--------|-------|
| Accuracy | 0.5600 |
| Weighted F1 | 0.5574 |
| RUL RMSE | 36.54 |
| C3 recall | 1.00 |

### Unified Classifier

| Subset | Accuracy | F1 | RMSE | C3 recall |
|--------|----------|----|------|-----------|
| FD001 | 0.4800 | 0.479 | 42.70 | 0.57 |
| FD002 | 0.6178 | 0.615 | 36.36 | 0.68 |
| FD003 | 0.4900 | 0.467 | 46.41 | 0.67 |
| FD004 | 0.5403 | 0.531 | 45.91 | 0.57 |

The unified classifier improves cross-dataset behavior substantially compared
with the FD001-only model, while trading off some FD001 specialization.

## Application

The local demo app is under `turbofan_app/`.

### Backend

`turbofan_app/backend/app.py` is a Flask API that loads the unified 1D-CNN and
serves:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Model/API health |
| `/api/classes` | GET | Class metadata and feature list |
| `/api/predict` | POST | CSV upload inference |
| `/api/sample/<class_id>` | GET | Demo sample response for class `0-3` |

The backend expects CSV uploads with the 16 active feature columns. Extra columns
are ignored.

### Frontend

`turbofan_app/frontend/` is a React/Vite dashboard with:

- CSV upload
- API health indicator
- class probability display
- estimated RUL display
- interactive Three.js turbofan viewer
- highlighted fault zones for C1-C3 predictions

The frontend currently uses `http://localhost:5000/api` as the API base URL.

## Repository Structure

```text
nasa_gan_turbo_fan_engine/
  BUGS.md
  IMPLEMENTATION.md
  README.md
  configs/
    data_config.yaml
    model_config.json
  data/
    raw/
    processed/
    synthetic/
  notebooks/
    01_data_quality_and_rul.ipynb
    ...
    13_unified_classifier_colab.ipynb
  reports/
    figures/
  src/
    ingestion/
    preprocessing/
    phase3_physics_labels/
  tests/
    engine_*.csv
  turbofan_app/
    backend/
    frontend/
```

## Known Limitations

- Synthetic windows match marginal distributions better than joint
  cross-sensor relationships.
- Upload-time scaling in the Flask API can fit a scaler on the uploaded window
  when full production scalers are unavailable.
- Training notebooks use fixed epoch counts; early stopping is still pending.
- The app is a local demo, not a hardened production deployment.

## Tech Stack

- Python
- pandas, NumPy, scikit-learn, SciPy, joblib
- PyTorch
- Jupyter notebooks and Google Colab for training
- Flask and Flask-CORS
- React, Vite, Three.js, lucide-react
