# Implementation Plan: cmapss-bug-fixes

## Overview

Fix all 11 active bugs in the NASA CMAPSS turbofan GAN pipeline. Changes span
two surfaces: `src/preprocessing/preprocess.py` (Bugs 001, 009, 010) and
six Jupyter notebooks 01–07 / 05b (Bugs 002, 004, 005, 006, 007, 008, 012, 013).
All fixes are additive or in-place edits — no new modules are introduced.

A pytest + Hypothesis test suite is created under `tests/` to validate the
pure-function changes in `preprocess.py` and the early-stopping control flow.

---

## Tasks

- [ ] 1. Set up test infrastructure
  - Create `tests/` directory with `__init__.py` and `conftest.py`
  - Add `pytest` and `hypothesis` to `requirements.txt` (or `pyproject.toml`) if not already present
  - Create empty `tests/test_preprocess.py` and `tests/test_early_stopping.py` stubs
  - _Requirements: none (scaffolding)_

- [ ] 2. Fix `preprocess.py` — BUG-001, BUG-009, BUG-010

  - [ ] 2.1 Refactor `normalize_per_engine` to return a `Scalers_Dict`
    - Change return type from `pd.DataFrame` to `tuple[pd.DataFrame, dict[int, MinMaxScaler]]`
    - Collect each fitted scaler into `scalers` dict keyed by `int(engine_id)`
    - Return `(scaled_df, scalers)` — keep the sort/reset_index logic unchanged
    - Update the `run_preprocessing` call site to unpack both values
    - _Requirements: 1.1_

  - [ ]* 2.2 Write property test for `normalize_per_engine` — Property 1 (Scalers_Dict completeness)
    - **Property 1: Scalers_Dict completeness**
    - Use `@given` with a strategy that generates DataFrames with 1–20 distinct `engine_id` values and at least 1 cycle per engine
    - Assert `set(scalers.keys()) == set(df["engine_id"].astype(int).unique())`
    - **Validates: Requirements 1.1**

  - [ ] 2.3 Implement `normalize_per_engine_apply`
    - New function: accepts `df: pd.DataFrame` and `scalers_path: Path`
    - Raises `FileNotFoundError` (with path) if `scalers_path` does not exist
    - Loads scalers via `joblib.load`; calls `scaler.transform` (no fit/fit_transform)
    - Raises `KeyError("Engine ID {eid} not found ...")` for any engine with no scaler entry
    - _Requirements: 1.3, 1.4, 1.5_

  - [ ]* 2.4 Write property test for `normalize_per_engine_apply` — Property 2 (values in unit range)
    - **Property 2: Scaler apply — values in unit range**
    - Fit scalers via `normalize_per_engine`, save to a temp file, then apply via `normalize_per_engine_apply` on the same DataFrame
    - Assert all values in `FEATURE_COLS` are in `[0.0, 1.0]` (allow float32 tolerance of 1e-6)
    - **Validates: Requirements 1.3**

  - [ ] 2.5 Add `joblib.dump` call in `run_preprocessing` to persist scalers
    - After the `normalize_per_engine` call, call `joblib.dump(scalers, output_dir / "scalers.pkl")`
    - Add `import joblib` at top of file
    - Print confirmation: `"Scalers saved : {path.resolve()}"`
    - _Requirements: 1.2_

  - [ ] 2.6 Add `subset` parameter and path-routing to `run_preprocessing`
    - Add `subset: str = "FD001"` parameter; validate against `VALID_SUBSETS = {"FD001","FD002","FD003","FD004"}`
    - Raise `ValueError` (listing valid names) for unknown subset values
    - Default `input_path` to `data/processed/{subset}/train_with_rul.csv` when `None`
    - Default `output_dir` to `data/processed/{subset}/` when `None`
    - Raise `FileNotFoundError` (with path) if `input_path` does not exist
    - Update the `if __name__ == "__main__"` block to call `run_preprocessing(subset="FD001")`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 2.7 Write property test for subset routing — Property 6
    - **Property 6: Subset-routing correctness**
    - For each valid subset in `{"FD001","FD002","FD003","FD004"}`, mock the file-system and assert that `run_preprocessing(subset=s)` targets only `data/processed/{s}/` paths
    - Assert `ValueError` is raised for any string outside the valid set (use `@given(st.text())` filtered to exclude valid names)
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

  - [ ] 2.8 Implement `preprocess_test_set`
    - New function with signature `preprocess_test_set(test_path, rul_path, scalers_path, output_dir, window_size=30)`
    - Raise `FileNotFoundError` (with path) if `scalers_path` does not exist
    - Load test CSV with `COLUMN_NAMES` schema (26 cols: `engine_id, cycle, op1–op3, s1–s21`); drop `DROP_SENSORS`
    - Call `normalize_per_engine_apply` (no refit)
    - Per engine: sort by cycle, extract last `window_size` rows; skip (warn) if engine has < 30 cycles
    - Save `X_test.npy` (shape `(N_valid, 30, 17)`, float32) and `y_test_rul.npy` (shape `(N_valid,)`, float32)
    - Create `output_dir` with `parents=True, exist_ok=True` before saving
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 2.9 Write property test for `preprocess_test_set` windowing — Property 5
    - **Property 5: Test windowing correctness and output alignment**
    - Use `@given` strategy generating per-engine cycle counts (mix of values ≥ 30 and < 30)
    - Assert `X_test.shape == (N_valid, 30, 17)` and `y_test_rul.shape == (N_valid,)`
    - Assert each `X_test[i]` exactly equals the last 30 rows of the matching engine's features
    - Assert engines with < 30 cycles are absent from output
    - **Validates: Requirements 8.4, 8.5, 8.6**

- [ ] 3. Checkpoint — preprocess.py complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Fix notebooks — seeds and figures directory (BUG-005, BUG-012)

  - [ ] 4.1 Add `SEED = 42` + `np.random.seed(SEED)` to Cell 1 of notebooks 01, 02, 03
    - Notebooks: `01_data_quality_and_rul.ipynb`, `02_preprocessing.ipynb`, `03_physics_labels.ipynb`
    - Insert the seed block as the very first lines of Cell 1 in each notebook
    - _Requirements: 4.1, 4.2_

  - [ ] 4.2 Add `SEED = 42` + numpy/torch seeds to Cell 1 of notebooks 04, 05, 05b, 06, 07
    - Notebooks: `04_model_architecture.ipynb`, `05_training.ipynb`, `05b_training_cgan.ipynb`, `06_synthetic_fault_generation.ipynb`, `07_validation.ipynb`
    - Insert `SEED = 42`, `np.random.seed(SEED)`, `torch.manual_seed(SEED)`, `torch.cuda.manual_seed_all(SEED)` as the first lines of Cell 1
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 4.3 Add `random_state=SEED` to all sklearn randomised calls in all notebooks
    - Locate every call to `RandomForestClassifier(...)` and `train_test_split(...)` across notebooks 01–07 and 05b
    - Add `random_state=SEED` to each call that does not already have it
    - _Requirements: 4.4_

  - [ ] 4.4 Add `reports/figures/` auto-creation to Cell 1 of notebooks 03, 04, 05, 05b, 06, 07
    - Append `from pathlib import Path` (if not already imported) and `Path("../reports/figures").mkdir(parents=True, exist_ok=True)` to Cell 1 of each listed notebook
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [ ]* 4.5 Write property test for deterministic pipeline functions — Property 4
    - **Property 4: Deterministic pipeline functions**
    - For any training DataFrame, seed numpy with `SEED=42`, call `normalize_per_engine` twice and assert identical scaled DataFrames
    - Call `make_windows` twice on the same DataFrame and assert identical `X` and `y` arrays
    - **Validates: Requirements 4.5**

- [ ] 5. Fix Phase 6 notebook — CGAN checkpoints and model classes (BUG-002, BUG-007)

  - [ ] 5.1 Update `MODELS_DIR` and `model_config.json` path in Cell 1 of `06_synthetic_fault_generation.ipynb`
    - Change `MODELS_DIR = Path("../data/processed/FD001/checkpoints")` to `MODELS_DIR = Path("../data/processed/FD001/checkpoints_cgan")`
    - Change the `open("../configs/model_config.json")` call to `open(MODELS_DIR / "model_config.json")`
    - Delete any secondary cell that overrides `MODELS_DIR` back to the TimeGAN path
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 5.2 Replace TimeGAN model class definitions with `CGANGenerator` and `CGANDiscriminator` in `06_synthetic_fault_generation.ipynb`
    - Remove the cell that defines `Embedder`, `Recovery`, `Supervisor`, `Generator`, `Discriminator` and their `load_state_dict` calls
    - Add a new cell that defines `CGANGenerator` and `CGANDiscriminator` (copy verbatim from `05b_training_cgan.ipynb` Cells 3–4)
    - _Requirements: 2.2, 2.3_

  - [ ] 5.3 Implement `_safe_load` helper and use it to load both CGAN components in `06_synthetic_fault_generation.ipynb`
    - Define `_safe_load(model, ckpt_path)` that inspects top-level state dict keys, raises `ValueError` (with expected vs actual key sets) on mismatch, then calls `model.load_state_dict(ckpt, strict=True)`
    - Call `_safe_load(G_net, MODELS_DIR / "generator.pt")` and `_safe_load(D_net, MODELS_DIR / "discriminator.pt")`
    - Set both models to `.eval()`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ] 5.4 Replace the `generate_samples` function in `06_synthetic_fault_generation.ipynb` with the CGAN version
    - Remove the TimeGAN generation function (calls `generator → supervisor → recovery`)
    - Add the CGAN `generate_samples(generator, n_samples, class_id, device)` that batches `torch.randn` noise + class conditioning
    - Update all downstream `generate_samples` calls to use `G_net` without TimeGAN arguments
    - _Requirements: 2.2, 2.3_

- [ ] 6. Fix Phase 6 notebook — dead-sensor rescaling and interpolation labels (BUG-004, BUG-006)

  - [ ] 6.1 Add dead-sensor post-processing rescaling cell in `06_synthetic_fault_generation.ipynb`
    - After the generation loop, add a cell that: flattens `X_train.npy` to `(N*30, 17)`, computes `real_mean` and `real_std`, defines and applies `rescale_to_real_stats(synth)` using the formula `x_rescaled = (x_raw - raw_mean)/(raw_std + 1e-8) * real_std + real_mean`, clips to `[0.0, 1.0]`
    - Overwrite `synth_X` with the rescaled result before saving `synth_X.npy`
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 6.2 Write property test for rescaling formula correctness — Property 3
    - **Property 3: Rescaling formula correctness**
    - Use `@given` with arrays of shape `(N, 30, 17)` where N ≥ 100 (generated via `hypothesis.extra.numpy`)
    - Assert all values in the output are in `[0.0, 1.0]`
    - For arrays where clipping does not occur at the mean, assert `|output_mean - real_mean| < 0.05` per feature
    - **Validates: Requirements 3.2, 3.4**

  - [ ] 6.3 Save interpolation label arrays in `06_synthetic_fault_generation.ipynb`
    - After saving `interp_C1_C2.npy`, add: `np.save(SYNTH_DIR / "interp_C1_C2_labels.npy", np.ones(len(interp_C1_C2), dtype=np.int64))`
    - After saving `interp_C2_C3.npy`, add: `np.save(SYNTH_DIR / "interp_C2_C3_labels.npy", np.full(len(interp_C2_C3), 2, dtype=np.int64))`
    - Ensure `SYNTH_DIR.mkdir(parents=True, exist_ok=True)` is called before the save block
    - _Requirements: 5.1, 5.2, 5.3_

- [ ] 7. Fix Phase 7 notebook — load CGAN data (BUG-008)

  - [ ] 7.1 Update data-loading cell in `07_validation.ipynb` to load from `Synth_Dir`
    - In Cell 1, define `SYNTH_DIR = Path("../data/synthetic/GAN")`
    - Add a guard: if `synth_X.npy` or `synth_labels.npy` does not exist, raise `FileNotFoundError` instructing the user to run Phase 6 first
    - Load `X_synth = np.load(SYNTH_DIR / "synth_X.npy")` and `y_synth = np.load(SYNTH_DIR / "synth_labels.npy")`
    - Remove all remaining references to `TimeGAN_Checkpoints_Dir` paths or TimeGAN-generated arrays
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ] 7.2 Update discriminative score and per-class metric cells in `07_validation.ipynb` to use `X_synth` and `y_synth`
    - Find every cell that passes a synthetic array to the discriminative score, MMD, or KS-test computation
    - Replace the TimeGAN-era variable names with `X_synth` / `y_synth`
    - Pass `random_state=SEED` to `RandomForestClassifier` (covered also by task 4.3 but verify here)
    - _Requirements: 7.4, 7.5_

- [ ] 8. Add early stopping to CGAN training (BUG-013)

  - [ ] 8.1 Implement `compute_discriminative_score` helper in `05b_training_cgan.ipynb`
    - Add imports for `RandomForestClassifier`, `cross_val_score`, `StandardScaler` in the relevant cell
    - Define `compute_discriminative_score(G_model, X_real, y_real, n_samples=500, seed=SEED)` following the design spec: stratified real sample, generate matching synthetic, flatten, standardise, 5-fold CV RF with `n_estimators=100, random_state=seed`
    - Return `scores.mean()`
    - _Requirements: 11.1_

  - [ ] 8.2 Replace the training loop in Cell 6 of `05b_training_cgan.ipynb` with the early-stopping version
    - Add `EARLY_STOP_THRESHOLD = 0.60` and `EVAL_INTERVAL = 50` constants
    - After each completed epoch, check `(epoch + 1) % EVAL_INTERVAL == 0`; if so, call `compute_discriminative_score`; if score < threshold, print the trigger message and `break`
    - After the loop, unconditionally save `generator.pt`, `discriminator.pt`, and `model_config.json` to `MODELS_DIR`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ]* 8.3 Write property test for early-stopping control flow — Property 7
    - **Property 7: Early stopping control flow**
    - Abstract the loop into a testable function `run_training_loop(epoch_scores: list[float]) -> int` that returns the exit epoch
    - Use `@given` with lists of floats representing per-50-epoch discriminative scores
    - Assert exit epoch equals the index of the first score < 0.60 (×50), or 500 if none are below threshold
    - Assert checkpoint-save is triggered in both paths
    - **Validates: Requirements 11.2, 11.3, 11.4, 11.5**

  - [ ]* 8.4 Write unit tests for early-stopping checkpoints
    - `test_checkpoints_saved_after_early_stop`: mock `torch.save` and `json.dump`; verify called after early-stop break
    - `test_checkpoints_saved_after_full_run`: same mocks; verify called after all 500 epochs
    - _Requirements: 11.4_

- [ ] 9. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/ -v` and confirm all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (Properties 1–7 from design)
- Unit tests validate specific examples and edge cases
- Notebook changes are JSON cell edits; tools like `nbformat` can be used for programmatic editing or cells can be edited directly in the Jupyter JSON
- The design's `_safe_load` helper (BUG-007) is implemented inline in task 5.3 alongside the CGAN loading code (BUG-002), since both touch the same notebook cell

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1"] },
    { "id": 1, "tasks": ["2.2", "2.3", "2.6"] },
    { "id": 2, "tasks": ["2.4", "2.5", "2.7", "2.8"] },
    { "id": 3, "tasks": ["2.9", "4.1", "4.2", "4.3"] },
    { "id": 4, "tasks": ["4.4", "4.5", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3"] },
    { "id": 6, "tasks": ["5.4", "6.1", "7.1"] },
    { "id": 7, "tasks": ["6.2", "6.3", "7.2", "8.1"] },
    { "id": 8, "tasks": ["8.2"] },
    { "id": 9, "tasks": ["8.3", "8.4"] }
  ]
}
```
