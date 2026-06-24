# NASA CMAPSS Turbofan Fault Classification

Synthetic-data-assisted fault classification for NASA CMAPSS turbofan engine
sensor data. The project converts RUL into degradation classes, trains
Conditional GANs to balance rare fault states, trains a 1D-CNN classifier, and
serves the trained model through a local Flask + React dashboard.

## What This Project Does

- Loads NASA CMAPSS FD001-FD004 train/test/RUL files.
- Builds 30-cycle sensor windows with 16 active features.
- Maps capped RUL into four fault classes: healthy, early wear, advanced fault,
  and imminent failure.
- Uses CGAN-generated windows to balance minority fault classes.
- Trains a unified 1D-CNN classifier across all four CMAPSS subsets.
- Provides a local web app for CSV upload inference and 3D fault visualization.

## Project Layout

```text
configs/                 Pipeline and model configuration
data/raw/                CMAPSS raw train/test/RUL files
data/processed/          Processed arrays, scalers, checkpoints, classifiers
data/synthetic/          CGAN-generated synthetic windows
notebooks/               End-to-end experimental pipeline
reports/figures/         Training, validation, and evaluation plots
src/                     Reusable ingestion/preprocessing/labeling helpers
tests/                   Sample engine CSV files for app testing
turbofan_app/backend/    Flask inference API
turbofan_app/frontend/   React/Vite dashboard with Three.js viewer
```

## Fault Classes

| Class | RUL range | Meaning |
|-------|-----------|---------|
| C0 | `> 100` | Healthy |
| C1 | `51-100` | Early wear |
| C2 | `11-50` | Advanced fault |
| C3 | `0-10` | Imminent failure |

## Model Results

Unified classifier results reported in the project docs:

| Subset | Accuracy | F1 | C3 recall |
|--------|----------|----|-----------|
| FD001 | 0.4800 | 0.479 | 0.57 |
| FD002 | 0.6178 | 0.615 | 0.68 |
| FD003 | 0.4900 | 0.467 | 0.67 |
| FD004 | 0.5403 | 0.531 | 0.57 |

## Run the API

From the backend folder:

```powershell
cd turbofan_app\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The API runs at:

```text
http://localhost:5000
```

Useful endpoints:

- `GET /api/health`
- `GET /api/classes`
- `POST /api/predict` with form field `file`
- `GET /api/sample/0`, `/api/sample/1`, `/api/sample/2`, `/api/sample/3`

## Run the Frontend

From the frontend folder:

```powershell
cd turbofan_app\frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://localhost:5173
```

Keep the Flask backend running on port `5000` so uploads and sample buttons can
reach the API.

## CSV Input Format

Uploaded CSV files must include these columns:

```text
op1, op2,
s2, s3, s4, s7, s8, s9,
s11, s12, s13, s14, s15, s17, s20, s21
```

The backend uses the last 30 rows. If fewer than 30 rows are provided, it pads
the front of the sequence with zeros. Extra columns are ignored.

Sample engine CSV files are available in `tests/`.

## Reproduce the Research Pipeline

The notebooks are ordered by phase:

```text
01_data_quality_and_rul.ipynb
02_preprocessing.ipynb
03_physics_labels.ipynb
04_model_architecture.ipynb
05_training.ipynb
05b_training_cgan.ipynb
06_synthetic_fault_generation.ipynb
07_validation.ipynb
08_classifier.ipynb
09_cross_dataset_eval.ipynb
10_multi_subset_preprocessing.ipynb
11_multi_subset_cgan_colab.ipynb
12_multi_subset_generation.ipynb
13_unified_classifier_colab.ipynb
```

GPU training was done in Colab for CGAN and classifier phases.

## Documentation

- `IMPLEMENTATION.md` explains the pipeline, models, app, and results.
- `BUGS.md` tracks resolved bugs, known limitations, and recommended next fixes.

## Current Limitations

- Synthetic data validation still shows weak KS pass rate despite good MMD.
- The local API has a demo-friendly fallback scaler for uploaded windows.
- Training loops use fixed epoch counts rather than early stopping.
- The frontend API URL is currently hard-coded to `http://localhost:5000/api`.
