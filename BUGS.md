# NASA CMAPSS — Known Bugs & Issues

## CRITICAL (must fix before Phase 8)

---

### BUG-001: Scaler not saved after Phase 2
**File:** `src/preprocessing/preprocess.py`, `02_preprocessing.ipynb`
**Severity:** Critical — causes data leakage
**Description:**
MinMaxScaler is fit per engine unit during preprocessing but never saved to disk.
When test_FD001.txt (and FD002-FD004) are preprocessed for Phase 8 evaluation,
a new scaler will be fit on test data — this is data leakage and invalidates results.
**Fix:**
```python
import joblib
scalers = {}
for engine_id, group in df.groupby("engine_id"):
    scaler = MinMaxScaler()
    scalers[engine_id] = scaler
joblib.dump(scalers, DATA_DIR / "scalers.pkl")
```
Apply saved scalers (not refit) when preprocessing test set.

---

### BUG-002: CGAN Phase 6 still uses TimeGAN checkpoints
**File:** `06_synthetic_fault_generation.ipynb` Cell 1
**Severity:** Critical — wrong model being used for generation
**Description:**
Phase 6 loads from `checkpoints/` (TimeGAN) not `checkpoints_cgan/` (CGAN).
All synthetic data currently saved is from the failed TimeGAN model.
**Fix:**
```python
# change in Phase 6 Cell 1
MODELS_DIR = Path("../data/processed/FD001/checkpoints_cgan")
```
Then rerun Phase 6 → 7 → 8 fully after CGAN training completes.

---

### BUG-003: INPUT_DIM mismatch hardcoded in model_config.json
**File:** `configs/model_config.json`
**Severity:** Critical — causes RuntimeError on load
**Description:**
model_config.json was initially written with INPUT_DIM=14 but actual
X_train.npy has shape (N, 30, 17). Was fixed in Phase 5 Cell 1 by
auto-detecting from X.shape[2] but old config files may still exist
in checkpoints directories with wrong value.
**Fix:**
Always derive INPUT_DIM from data:
```python
INPUT_DIM = X.shape[2]  # never hardcode
```
Delete and regenerate any config files that have `"input_dim": 14`.

---

## HIGH (fix before final submission)

---

### BUG-004: Dead sensors in synthetic output (s3, op2)
**File:** `05_training_timegan.ipynb`, `05b_training_cgan.ipynb`
**Severity:** High — reduces synthetic data quality
**Description:**
s3 (fan speed Nf) and op2 consistently output near-zero values across
all generated classes. These sensors are near-zero after per-engine
MinMaxScaling for some engines, causing the generator to learn zero
as the default output for those features.
**Root cause:** Per-engine scaling means some engines have s3 range
very close to zero, creating a multi-modal distribution that the
generator collapses to the mode at zero.
**Fix options:**
- Scale globally across all engines for these specific sensors
- Add a feature-wise variance penalty to generator loss
- Post-process: rescale synthetic outputs to match real feature statistics

---

### BUG-005: No reproducibility seed set
**File:** All notebooks
**Severity:** High — results not reproducible
**Description:**
No random seed is set for numpy, torch, or sklearn operations.
Every run produces different synthetic data and potentially different
validation scores.
**Fix:** Add to Cell 1 of every notebook:
```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

### BUG-006: Interpolated samples have no class label
**File:** `06_synthetic_fault_generation.ipynb` Cell 8
**Severity:** High — interpolated data unusable for classifier
**Description:**
interp_C1_C2.npy and interp_C2_C3.npy are generated but saved without
corresponding label arrays. They cannot be used in Phase 8 classifier
training without labels.
**Fix:**
Assign soft labels (e.g. 1.5 for C1-C2 midpoint) or hard labels
(round to nearest class) and save alongside:
```python
np.save(SYNTH_DIR / "interp_C1_C2_labels.npy",
        np.full(len(interp_C1_C2), 1))   # label as C1
np.save(SYNTH_DIR / "interp_C2_C3_labels.npy",
        np.full(len(interp_C2_C3), 2))   # label as C2
```

---

### BUG-007: TimeGAN checkpoint architecture mismatch
**File:** `05_training_timegan.ipynb`, `06_synthetic_fault_generation.ipynb`
**Severity:** High — causes RuntimeError on load_state_dict
**Description:**
Recovery architecture was changed across training iterations
(Sequential+Sigmoid → Linear → Linear+clamp) causing checkpoint
load failures when Phase 6 tries to load Phase 5 weights.
**Fix:**
Always inspect checkpoint keys before loading:
```python
ckpt = torch.load("recovery.pt", map_location="cpu")
print(list(ckpt.keys()))
```
Match class definition to key names exactly.
Document final architecture in model_config.json.

---

## MEDIUM (improve quality)

---

### BUG-008: Phase 7 validation run on TimeGAN data only
**File:** `07_validation.ipynb`
**Severity:** Medium — validation results not yet meaningful
**Description:**
All Phase 7 metrics (MMD, KS, discriminative score) were computed on
TimeGAN synthetic data which was known to be poor quality.
Phase 7 needs to be rerun after CGAN generation completes.

---

### BUG-009: test_FD001.txt never preprocessed
**File:** Phase 8 not yet implemented
**Severity:** Medium — Phase 8 cannot run without this
**Description:**
test_FD001.txt and RUL_FD001.txt have never been loaded or preprocessed.
Phase 8 requires the test set to evaluate classifier performance.
**Fix needed in Phase 8 Cell 1:**
- Load test_FD001.txt
- Apply saved scalers from BUG-001 fix (not refit)
- Apply sliding window W=30
- For each engine take the last window only
- Compare predicted RUL vs RUL_FD001.txt ground truth

---

### BUG-010: FD002/FD003/FD004 not yet used
**File:** N/A — future work
**Severity:** Medium — limits generalization claim
**Description:**
Only FD001 has been processed. Cross-dataset evaluation on FD002-FD004
is planned for Phase 9 but preprocessing pipeline has not been extended.
**Fix:** Parameterize preprocessing scripts by subset name:
```python
for subset in ["FD001", "FD002", "FD003", "FD004"]:
    run_preprocessing(subset)
```

---

## LOW (nice to fix)

---

### BUG-011: Unicode arrow in print statement
**File:** `src/phase3_physics_labels/labeler.py` line 49
**Severity:** Low — crashes on Windows cp1252 encoding
**Description:** `→` character causes UnicodeEncodeError on Windows terminals.
**Fix:** Replace `→` with `->` in all print statements.

---

### BUG-012: reports/figures/ directory not auto-created
**File:** All notebooks that call plt.savefig()
**Severity:** Low — FileNotFoundError if directory missing
**Fix:** Add to Cell 1 of notebooks 03 onwards:
```python
Path("../reports/figures").mkdir(parents=True, exist_ok=True)
```

---

### BUG-013: No early stopping in CGAN training
**File:** `05b_training_cgan.ipynb` Cell 6
**Severity:** Low — wastes compute if model converges early
**Description:** Training runs full 500 epochs regardless of convergence.
**Fix:** Track discriminative score every 50 epochs, stop if < 0.60.

---

## Summary

| Priority | Count | Status         |
|----------|-------|----------------|
| Critical | 3     | Unresolved     |
| High     | 4     | Unresolved     |
| Medium   | 3     | Planned        |
| Low      | 3     | Nice to have   |
| **Total**| **13**|                |

**Immediate priority order:**
1. BUG-002 — point Phase 6 to CGAN checkpoints after 05b training
2. BUG-001 — save scaler before Phase 8
3. BUG-005 — add seeds everywhere
4. BUG-006 — add labels to interpolated samples
