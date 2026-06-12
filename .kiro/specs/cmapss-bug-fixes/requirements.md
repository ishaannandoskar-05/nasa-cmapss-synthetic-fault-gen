# Requirements Document

## Introduction

This document specifies requirements to fix all known bugs in the NASA CMAPSS turbofan engine GAN project. The project generates synthetic fault data for supervised classifier training using a Conditional GAN (CGAN) on NASA CMAPSS sensor data across an 8-phase Jupyter notebook pipeline. BUG-003 and BUG-011 are already resolved and are excluded from this specification.

The bugs to address are:

| ID | Severity | Title |
|----|----------|-------|
| BUG-001 | Critical | Scaler not saved after Phase 2 |
| BUG-002 | Critical | CGAN Phase 6 still uses TimeGAN checkpoints |
| BUG-004 | High | Dead sensors in synthetic output (s3, op2) |
| BUG-005 | High | No reproducibility seed set |
| BUG-006 | High | Interpolated samples have no class label |
| BUG-007 | High | TimeGAN checkpoint architecture mismatch |
| BUG-008 | Medium | Phase 7 validation run on TimeGAN data only |
| BUG-009 | Medium | test_FD001.txt never preprocessed |
| BUG-010 | Medium | FD002/FD003/FD004 not yet used |
| BUG-012 | Low | reports/figures/ directory not auto-created |
| BUG-013 | Low | No early stopping in CGAN training |

---

## Glossary

- **Pipeline**: The multi-phase Jupyter notebook pipeline (Phases 1–8) used to generate synthetic fault data.
- **Preprocessing_Module**: The Python module at `src/preprocessing/preprocess.py`.
- **Scaler**: A `MinMaxScaler` instance from scikit-learn, fitted per engine unit to normalize sensor features over `FEATURE_COLS`.
- **Scalers_Dict**: A Python dictionary keyed by integer engine ID mapping to the corresponding fitted `Scaler` object.
- **FEATURE_COLS**: The list of feature columns used for scaling: `["op1", "op2", "op3", "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]` (3 op cols + 14 sensor cols after dropping DROP_SENSORS).
- **CGAN**: Conditional Generative Adversarial Network implemented in `05b_training_cgan.ipynb`, comprising `CGANGenerator` and `CGANDiscriminator`.
- **TimeGAN**: An earlier Time-series GAN model (rejected in favour of CGAN). Its checkpoints reside in `data/processed/FD001/checkpoints/`.
- **CGAN_Checkpoints_Dir**: The directory `data/processed/FD001/checkpoints_cgan/`.
- **TimeGAN_Checkpoints_Dir**: The directory `data/processed/FD001/checkpoints/`.
- **Synth_Dir**: The directory `data/synthetic/GAN/`.
- **Interp_Sample**: A synthetic time-series window produced by interpolating between two class embeddings.
- **N_INTERP**: The number of interpolated samples generated per pair; equals 500.
- **Discriminative_Score**: A random-forest-based metric used in validation (Phase 7) to quantify how distinguishable synthetic samples are from real samples. A score near 0.50 is ideal.
- **Dead_Sensor**: A feature column whose generated values are near-zero across all classes due to per-engine MinMaxScaling collapsing multi-modal distributions.
- **SEED**: An integer constant (42) used as a fixed random seed for all stochastic operations.
- **Window**: A fixed-length (W=30 cycles) time-series segment extracted from engine telemetry.
- **Feature_Stats**: Per-feature mean and standard deviation computed over real training windows by flattening `X_train.npy` from shape `(N, 30, 17)` to `(N*30, 17)` before computing statistics.
- **Test_Set**: The test portion of the CMAPSS dataset (`test_FD001.txt`) paired with ground-truth RUL values (`RUL_FD001.txt`).
- **Subset**: One of the four CMAPSS dataset partitions: FD001, FD002, FD003, FD004.
- **Reports_Figures_Dir**: The directory `reports/figures/` (relative to the project root; `../reports/figures/` relative to notebooks).

---

## Requirements

### Requirement 1: BUG-001 — Save and Apply Scalers (Data Leakage Prevention)

**User Story:** As a data scientist, I want scalers fitted during training-set preprocessing to be persisted to disk, so that test-set preprocessing can apply the same transformations without refitting and thereby avoid data leakage.

#### Acceptance Criteria

1. WHEN `normalize_per_engine` is called with a training `DataFrame`, THE `Preprocessing_Module` SHALL fit one `Scaler` over `FEATURE_COLS` per engine unit and collect all fitted scalers into a `Scalers_Dict` keyed by engine ID.
2. WHEN the training preprocessing run completes without error, THE `Preprocessing_Module` SHALL serialize the `Scalers_Dict` to `data/processed/{subset}/scalers.pkl` using `joblib.dump`, where `{subset}` is the value passed to `run_preprocessing`.
3. THE `Preprocessing_Module` SHALL expose a function named `normalize_per_engine_apply` that accepts a `DataFrame` and a path to a saved `Scalers_Dict` file, and returns a `DataFrame` of the same shape with `FEATURE_COLS` transformed.
4. WHEN `normalize_per_engine_apply` is invoked, THE `Preprocessing_Module` SHALL load the serialized `Scalers_Dict` and transform each engine group's `FEATURE_COLS` using its pre-fitted `Scaler` without calling `fit` or `fit_transform`. IF an engine ID in the input `DataFrame` has no matching key in the loaded `Scalers_Dict`, THEN THE `Preprocessing_Module` SHALL raise a `KeyError` with a message identifying the missing engine ID.
5. IF `normalize_per_engine_apply` is called and the `Scalers_Dict` file does not exist at the specified path, THEN THE `Preprocessing_Module` SHALL raise a `FileNotFoundError` whose message includes the missing file path.

---

### Requirement 2: BUG-002 — Point Phase 6 to CGAN Checkpoints

**User Story:** As a ML engineer, I want Phase 6 synthetic generation to load from the CGAN checkpoint directory, so that generated synthetic data reflects the trained CGAN model rather than the discarded TimeGAN model.

#### Acceptance Criteria

1. WHEN notebook `06_synthetic_fault_generation.ipynb` is executed, THE `Pipeline` SHALL set `MODELS_DIR` to `Path("../data/processed/FD001/checkpoints_cgan")`.
2. THE `Pipeline` SHALL NOT reference any path that resolves to `TimeGAN_Checkpoints_Dir` (`data/processed/FD001/checkpoints/`) for loading model weights.
3. WHEN loading model weights in Phase 6, THE `Pipeline` SHALL instantiate and load only the `CGANGenerator` and `CGANDiscriminator` components using the architecture defined in `05b_training_cgan.ipynb`, reading hyperparameters from `CGAN_Checkpoints_Dir/model_config.json` rather than from `configs/model_config.json`.

---

### Requirement 3: BUG-004 — Rescale Dead Sensors in Synthetic Output

**User Story:** As a data scientist, I want the synthetic output's feature distributions to match the real data's statistics, so that dead sensors (s3, op2) do not bias downstream classifier training.

#### Acceptance Criteria

1. WHEN synthetic samples are generated by the CGAN generator in `06_synthetic_fault_generation.ipynb`, THE `Pipeline` SHALL compute `Feature_Stats` (per-feature mean and standard deviation) from `X_train.npy` by flattening the array from shape `(N, 30, 17)` to `(N*30, 17)` before computing statistics.
2. THE `Pipeline` SHALL apply a post-processing rescaling step to all synthetic windows that shifts and scales each of the 17 features so that the synthetic mean and standard deviation match the corresponding values in `Feature_Stats`. The rescaling formula is: `x_rescaled = (x_raw - x_raw.mean()) / (x_raw.std() + eps) * real_std + real_mean`, where `eps = 1e-8`.
3. WHEN the rescaling is complete, THE `Pipeline` SHALL clip all rescaled synthetic values to `[0, 1]` to preserve the MinMaxScaled domain, and SHALL overwrite `Synth_Dir/synth_X.npy` with the rescaled array.
4. FOR ALL 17 feature columns in the rescaled synthetic output, the absolute difference between the synthetic feature mean and the real feature mean SHALL be less than 0.05 when measured over at least 500 generated samples per class.

---

### Requirement 4: BUG-005 — Set Reproducibility Seeds in All Notebooks

**User Story:** As a ML engineer, I want every notebook to set fixed random seeds at startup, so that all runs produce identical synthetic data and validation scores given the same inputs.

#### Acceptance Criteria

1. THE `Pipeline` SHALL define `SEED = 42` as the first constant in Cell 1 of each of the following notebooks: `01_data_quality_and_rul.ipynb`, `02_preprocessing.ipynb`, `03_physics_labels.ipynb`, `04_model_architecture.ipynb`, `05_training.ipynb`, `05b_training_cgan.ipynb`, `06_synthetic_fault_generation.ipynb`, `07_validation.ipynb`.
2. WHEN `SEED` is defined, THE `Pipeline` SHALL call `np.random.seed(SEED)` in Cell 1 of each notebook.
3. WHERE PyTorch is imported in a notebook, THE `Pipeline` SHALL additionally call `torch.manual_seed(SEED)` and `torch.cuda.manual_seed_all(SEED)` in Cell 1 of that notebook.
4. WHERE `sklearn` operations involving randomness are used (e.g., `RandomForestClassifier`, `train_test_split`), THE `Pipeline` SHALL pass `random_state=SEED` to those calls.
5. FOR ALL runs executed with the same input data and `SEED = 42`, the synthetic arrays `synth_X.npy` and `synth_labels.npy` produced by Phase 6 SHALL be bit-for-bit identical across consecutive executions on the same hardware and library versions.

---

### Requirement 5: BUG-006 — Save Labels Alongside Interpolated Samples

**User Story:** As a data scientist, I want interpolated synthetic samples to include class-label arrays, so that they can be incorporated into Phase 8 classifier training without ambiguity.

#### Acceptance Criteria

1. WHEN `interp_C1_C2.npy` (shape `(N_INTERP, 30, 17)`, dtype `float32`) is saved in `06_synthetic_fault_generation.ipynb`, THE `Pipeline` SHALL also save `interp_C1_C2_labels.npy` to `Synth_Dir` as a `numpy` array of shape `(N_INTERP,)`, dtype `int64`, where every element equals `1` (hard label for class C1).
2. WHEN `interp_C2_C3.npy` (shape `(N_INTERP, 30, 17)`, dtype `float32`) is saved, THE `Pipeline` SHALL also save `interp_C2_C3_labels.npy` to `Synth_Dir` as a `numpy` array of shape `(N_INTERP,)`, dtype `int64`, where every element equals `2` (hard label for class C2).
3. IF `Synth_Dir` does not exist when saving interpolated label arrays, THEN THE `Pipeline` SHALL create it (with `parents=True, exist_ok=True`) before saving.

---

### Requirement 6: BUG-007 — Safe Checkpoint Loading with Key Inspection

**User Story:** As a ML engineer, I want checkpoint loading to validate architecture keys before calling `load_state_dict`, so that architecture mismatches produce informative error messages rather than cryptic `RuntimeError` exceptions.

#### Acceptance Criteria

1. WHEN a checkpoint file is loaded in `06_synthetic_fault_generation.ipynb`, THE `Pipeline` SHALL inspect the set of top-level keys of the loaded state dictionary before passing it to `load_state_dict`.
2. IF the set of keys of the loaded state dictionary does not exactly match the set of keys returned by `model.state_dict()` for the target model, THEN THE `Pipeline` SHALL raise a `ValueError` whose message contains both the expected key set and the actual key set found in the checkpoint file.
3. IF the key sets match exactly, THE `Pipeline` SHALL proceed with `model.load_state_dict(checkpoint, strict=True)` without raising an error.
4. THE key inspection SHALL be performed independently for both CGAN components: `CGANGenerator` and `CGANDiscriminator`.

---

### Requirement 7: BUG-008 — Run Phase 7 Validation Against CGAN Data

**User Story:** As a data scientist, I want Phase 7 validation metrics to be computed on CGAN-generated synthetic data, so that reported quality metrics reflect the model actually used for downstream tasks.

#### Acceptance Criteria

1. WHEN `07_validation.ipynb` is executed, THE `Pipeline` SHALL load `X_synth` from `Synth_Dir/synth_X.npy` and `y_synth` from `Synth_Dir/synth_labels.npy`.
2. IF `Synth_Dir/synth_X.npy` or `Synth_Dir/synth_labels.npy` does not exist when `07_validation.ipynb` is executed, THEN THE `Pipeline` SHALL raise a `FileNotFoundError` with a message instructing the user to run Phase 6 first.
3. THE `Pipeline` SHALL NOT reference any file path that resolves to `TimeGAN_Checkpoints_Dir` or to synthetic data files known to be generated by the TimeGAN model.
4. WHEN computing the `Discriminative_Score`, THE `Pipeline` SHALL use the loaded CGAN-generated `X_synth` array.
5. WHEN computing MMD and KS-test metrics per class, THE `Pipeline` SHALL use the loaded CGAN-generated `X_synth` array, subsetting by `y_synth` class labels.

---

### Requirement 8: BUG-009 — Implement Test-Set Preprocessing

**User Story:** As a data scientist, I want a function that preprocesses the held-out test set using the saved scalers and sliding-window logic, so that Phase 8 can evaluate the classifier on properly prepared test data.

#### Acceptance Criteria

1. THE `Preprocessing_Module` SHALL expose a function named `preprocess_test_set` that accepts four arguments: the path to `test_FD001.txt`, the path to `RUL_FD001.txt`, the path to a saved `Scalers_Dict` file, and an output directory path.
2. WHEN `preprocess_test_set` is invoked, THE `Preprocessing_Module` SHALL load `test_FD001.txt` as whitespace-separated data with no header, assigning the same 26-column schema used by the training loader (engine_id, cycle, op1–op3, s1–s21), and SHALL drop `DROP_SENSORS` to yield 17 features in `FEATURE_COLS`.
3. WHEN scaling test data, THE `Preprocessing_Module` SHALL apply `normalize_per_engine_apply` using the saved training-set scalers, without refitting any scaler on the test data.
4. WHEN windowing the test data, THE `Preprocessing_Module` SHALL apply a sliding window of size W=30 with stride=1 per engine and retain only the LAST complete window for each engine. IF an engine has fewer than 30 cycles in the test set, THEN THE `Preprocessing_Module` SHALL skip that engine and log a warning identifying the engine ID and its cycle count.
5. WHEN writing results, THE `Preprocessing_Module` SHALL create `output_dir` if it does not exist, then save `X_test.npy` with shape `(N_valid_engines, 30, 17)` where `N_valid_engines` is the count of engines that had at least 30 cycles.
6. WHEN writing results, THE `Preprocessing_Module` SHALL save ground-truth RUL values from `RUL_FD001.txt` aligned to `N_valid_engines` as `y_test_rul.npy` with shape `(N_valid_engines,)` and dtype `float32`.
7. IF `preprocess_test_set` is called and the saved `Scalers_Dict` file does not exist, THEN THE `Preprocessing_Module` SHALL raise a `FileNotFoundError` whose message includes the missing file path.

---

### Requirement 9: BUG-010 — Parameterize Preprocessing by Subset Name

**User Story:** As a data scientist, I want the preprocessing pipeline to accept a subset identifier (FD001–FD004), so that the same code can process all four CMAPSS dataset partitions without modification.

#### Acceptance Criteria

1. THE `Preprocessing_Module`'s `run_preprocessing` function SHALL accept a `subset` parameter whose valid values are the strings `"FD001"`, `"FD002"`, `"FD003"`, and `"FD004"`.
2. WHEN `run_preprocessing` is called with a valid `subset` value, THE `Preprocessing_Module` SHALL set `input_path` to `data/processed/{subset}/train_with_rul.csv`, set `output_dir` to `data/processed/{subset}/`, and create `output_dir` with `parents=True, exist_ok=True` before writing any output files.
3. WHEN `run_preprocessing` is called with a `subset` value, THE `Preprocessing_Module` SHALL save the fitted `Scalers_Dict` to `data/processed/{subset}/scalers.pkl`.
4. IF `run_preprocessing` is called with a `subset` value not in `{"FD001", "FD002", "FD003", "FD004"}`, THEN THE `Preprocessing_Module` SHALL raise a `ValueError` with a message listing the four valid subset names.
5. IF `run_preprocessing` is called without providing the `subset` argument, THEN THE `Preprocessing_Module` SHALL behave identically to being called with `subset="FD001"`.
6. IF `run_preprocessing` is called with a valid `subset` value and `data/processed/{subset}/train_with_rul.csv` does not exist, THEN THE `Preprocessing_Module` SHALL raise a `FileNotFoundError` whose message includes the missing file path.

---

### Requirement 10: BUG-012 — Auto-Create reports/figures/ Directory

**User Story:** As a notebook user, I want the figures output directory to be created automatically at notebook startup, so that `plt.savefig()` calls do not raise `FileNotFoundError` when the directory is absent.

#### Acceptance Criteria

1. WHEN Cell 1 of `03_physics_labels.ipynb` is executed, THE `Pipeline` SHALL ensure `Reports_Figures_Dir` exists and is writable before any figure-saving cell in that notebook is executed.
2. WHEN Cell 1 of `04_model_architecture.ipynb` is executed, THE `Pipeline` SHALL ensure `Reports_Figures_Dir` exists and is writable before any figure-saving cell in that notebook is executed.
3. WHEN Cell 1 of `05_training.ipynb` is executed, THE `Pipeline` SHALL ensure `Reports_Figures_Dir` exists and is writable before any figure-saving cell in that notebook is executed.
4. WHEN Cell 1 of `05b_training_cgan.ipynb` is executed, THE `Pipeline` SHALL ensure `Reports_Figures_Dir` exists and is writable before any figure-saving cell in that notebook is executed.
5. WHEN Cell 1 of `06_synthetic_fault_generation.ipynb` is executed, THE `Pipeline` SHALL ensure `Reports_Figures_Dir` exists and is writable before any figure-saving cell in that notebook is executed.
6. WHEN Cell 1 of `07_validation.ipynb` is executed, THE `Pipeline` SHALL ensure `Reports_Figures_Dir` exists and is writable before any figure-saving cell in that notebook is executed.
7. IF `Reports_Figures_Dir` already exists when Cell 1 is executed, THEN THE `Pipeline` SHALL proceed without raising any error.
8. IF directory creation fails due to a permission or I/O error, THEN THE `Pipeline` SHALL raise the underlying OS error immediately in Cell 1, before any `plt.savefig()` call is reached.

---

### Requirement 11: BUG-013 — Add Early Stopping to CGAN Training

**User Story:** As a ML engineer, I want CGAN training to stop automatically when the model has converged, so that compute time is not wasted on epochs that yield no further improvement.

#### Acceptance Criteria

1. WHEN `05b_training_cgan.ipynb` Cell 6 runs the training loop, THE `Pipeline` SHALL evaluate the `Discriminative_Score` at every 50th epoch (i.e., epochs 50, 100, 150, …, 500) using a class-stratified random sample of at least 500 real windows and 500 freshly generated synthetic windows, pre-processed with `StandardScaler` and evaluated with a `RandomForestClassifier(n_estimators=100, random_state=SEED)` using 5-fold cross-validation, consistent with the Phase 7 evaluation protocol.
2. IF the `Discriminative_Score` falls below `0.60` at any evaluation checkpoint, THEN THE `Pipeline` SHALL complete the current epoch, print a message to the notebook cell output indicating the epoch number and the score that triggered early stopping, and exit the training loop.
3. IF the `Discriminative_Score` never drops below `0.60`, THEN THE `Pipeline` SHALL complete all 500 epochs without early stopping.
4. WHEN the training loop exits (either via early stopping or by completing all 500 epochs), THE `Pipeline` SHALL save the `CGANGenerator` and `CGANDiscriminator` state dictionaries and the current `model_config.json` to `CGAN_Checkpoints_Dir`.
5. THE `Discriminative_Score` evaluation SHALL be skipped for epochs before the first multiple of 50, so that the first evaluation occurs at epoch 50.
