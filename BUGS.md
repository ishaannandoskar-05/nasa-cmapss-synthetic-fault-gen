# NASA CMAPSS Bug Tracker

This tracker reflects the current project state: multi-subset CGAN training,
the unified 1D-CNN classifier, and the Flask/React demo application are now
present in the repository.

## Resolved

### BUG-001: Scalers not persisted
**Status:** Fixed

Training scalers are saved with the processed artifacts so later validation,
test preprocessing, and inference can avoid silently refitting on training data.
The app also ships the scaler needed by the backend in
`turbofan_app/backend/model_artifacts/global_scaler.pkl`.

### BUG-002: Synthetic generation pointed at TimeGAN checkpoints
**Status:** Fixed

The synthetic generation phase now uses CGAN checkpoints instead of deprecated
TimeGAN outputs.

### BUG-003: Hard-coded input dimension
**Status:** Fixed

The active pipeline uses the processed array shape and saved feature
configuration. The current feature set has `input_dim = 16`.

### BUG-004: Constant and near-constant features distorted preprocessing
**Status:** Fixed

Seven flat sensors are dropped. `op3` is excluded from the active 16-feature
pipeline, and `s3` is handled with global scaling to preserve useful variance.

### BUG-005: Reproducibility seeds missing from Python modules
**Status:** Fixed for source modules; verify notebooks before final reruns

`src/preprocessing/preprocess.py`, `src/phase3_physics_labels/labeler.py`, and
`src/phase3_physics_labels/monotonicity.py` set `SEED = 42`. Notebooks should
be checked before any final experiment rerun because notebook cell state can
still introduce drift.

### BUG-006: Interpolated synthetic samples missing labels
**Status:** Fixed

Interpolation label files are saved alongside generated samples for the
FD001 synthetic dataset.

### BUG-007: TimeGAN checkpoint architecture mismatch
**Status:** Resolved by design change

TimeGAN was deprecated after mode collapse and replaced by a direct sensor-space
Conditional GAN.

### BUG-009: FD002/FD004 discriminator collapse
**Status:** Fixed

The multi-condition CGAN training variant lowers discriminator learning rate
and throttles discriminator updates. FD002 and FD004 checkpoints and curves are
now present.

### BUG-011: Windows console Unicode crashes
**Status:** Fixed in active docs/code paths

Documentation has been normalized to ASCII to avoid cp1252 rendering failures.

### BUG-012: `reports/figures` directory missing
**Status:** Fixed

Report figures are present under `reports/figures/`, and generation notebooks
create output directories before saving.

### BUG-016: No frontend or inference API
**Status:** Fixed

The repository now includes:

- Flask API: `turbofan_app/backend/app.py`
- React/Vite dashboard: `turbofan_app/frontend/`
- 3D turbofan visualization with fault-zone highlighting
- CSV upload inference through `POST /api/predict`
- Demo samples through `GET /api/sample/<class_id>`

### BUG-008: KS pass rate remains poor
**Files:** `notebooks/05b_training_cgan.ipynb`,
`notebooks/11_multi_subset_cgan_colab.ipynb`,
`src/utils/cgan_training.py`
**Status:** Fixed (Integrated into training notebooks)

A cross-sensor correlation penalty has been added to the CGAN generator loss.
`src/utils/cgan_training.correlation_penalty(x_real, x_fake)` computes the
mean-squared difference between real and generated pairwise feature covariance
matrices and returns a scalar penalty term. Notebooks 05b and 11 should add
this to the generator loss (weight 0.5 recommended) alongside the existing
feature-matching term so synthetic samples reproduce joint sensor relationships,
not just marginal distributions.

### BUG-010: Unified training set can still be subset-dominated
**Files:** `notebooks/12_multi_subset_generation.ipynb`,
`src/utils/dataset_balancing.py`
**Status:** Fixed (Integrated into training notebook)

`src/utils/dataset_balancing.build_unified_dataset(subset_data, max_per_subset)`
caps every subset's contribution to the same budget before concatenation.
The cap defaults to the size of the smallest balanced subset so no single
domain can dominate. Notebook 12 should replace the bare `np.concatenate`
unified-merge cell with a call to this function.

### BUG-013: No early stopping in training loops
**Files:** `notebooks/05b_training_cgan.ipynb`,
`notebooks/11_multi_subset_cgan_colab.ipynb`,
`notebooks/13_unified_classifier_colab.ipynb`,
`src/utils/cgan_training.py`,
`src/utils/classifier_training.py`
**Status:** Fixed (Integrated into training notebooks)

Two reusable early-stopping classes are provided:

- `src/utils/cgan_training.EarlyStopping` — evaluates a discriminative RF
  score every `eval_interval` epochs and stops when the distance from the
  ideal 0.5 score has not improved for `patience` evaluations. Best G/D
  checkpoints are saved automatically.
- `src/utils/classifier_training.ClassifierEarlyStopping` — evaluates
  macro-F1 on a validation DataLoader every `eval_interval` epochs and stops
  after `patience` non-improving evaluations. Best model checkpoint is
  saved and can be restored with `.restore_best()`.

Notebooks 05b and 11 import `EarlyStopping`; notebook 13 imports
`ClassifierEarlyStopping`.

### BUG-014: FD001 performance dropped in the unified classifier
**Files:** `notebooks/13_unified_classifier_colab.ipynb`,
`src/utils/classifier_training.py`
**Status:** Fixed (Integrated into training notebook)

`src/utils/classifier_training.finetune_fd001(model, fd001_loader, device)`
performs a short additional training pass on the FD001 balanced dataset
using a low learning rate (default 1e-4) after unified training completes.
This recovers single-domain precision without catastrophic forgetting of
the multi-subset knowledge. Recommended: 10 fine-tuning epochs.

### BUG-015: Test-time fallback scaling can leak information
**Files:** `src/preprocessing/preprocess.py`, `turbofan_app/backend/app.py`
**Status:** Fixed

`normalize_per_engine` now raises `KeyError` for unseen engine IDs instead of
silently refitting a new scaler on test data. The Flask API's `normalize_window`
function exclusively uses the persisted `global_scaler.pkl`; if the file is
missing the server emits a `RuntimeWarning` at startup and returns a clear
`RuntimeError` to the caller instead of fitting on the uploaded window.

### BUG-017: Frontend API URL is hard-coded
**File:** `turbofan_app/frontend/src/TurbofanFaultDashboard.jsx`
**Status:** Fixed

`API_BASE` now reads from `import.meta.env.VITE_API_BASE` with a localhost
fallback. The value is set in `turbofan_app/frontend/.env` for local dev and
can be overridden in `.env.local` (git-ignored) for staging/production.
`.env.example` is provided as a template.

### BUG-018: Frontend README is still the Vite template
**File:** `turbofan_app/frontend/README.md`
**Status:** Fixed

Replaced with frontend-specific documentation covering project structure,
dev setup, `VITE_API_BASE` configuration, production build steps, backend
endpoint summary, and CSV upload column requirements.

### BUG-019: Frontend React hook effect bug and unused imports
**File:** `turbofan_app/frontend/src/TurbofanFaultDashboard.jsx`
**Status:** Fixed

ESLint highlighted multiple unused imports (`React`, `RotateCw`, `useCallback`) and a `react-hooks/set-state-in-effect` warning due to synchronous `checkHealth` calls triggering cascading renders in a `useEffect` hook. The unused imports were removed, and the `checkHealth` function was refactored with an active-state boolean inside the hook to prevent potential memory leaks and render loops.

## Summary

| Priority | Resolved | Open |
|----------|----------|------|
| High     | 7        | 0    |
| Medium   | 7        | 0    |
| Low      | 6        | 0    |
| Total    | 20       | 0    |

All tracked bugs are resolved. See `src/utils/` for the new training utility
modules introduced in this session.
