# Design Document — cmapss-bug-fixes

## Overview

This document describes the implementation design for fixing all 11 known bugs in
the NASA CMAPSS turbofan engine GAN project. The project is an 8-phase Jupyter
notebook pipeline that generates synthetic fault data using a Conditional GAN (CGAN)
on CMAPSS sensor telemetry. BUG-003 and BUG-011 are already resolved and excluded.

The fixes span two primary surfaces:

- **`src/preprocessing/preprocess.py`** — modified to save/apply scalers, support
  multiple subsets, and expose a test-set preprocessing function.
- **Notebooks** — targeted cell-level edits to notebooks 01–07 and 05b to add seeds,
  fix checkpoint paths, add interpolation labels, safe loading, validation data
  sources, figure-directory creation, and early stopping.

No new Python modules are introduced. All changes are additive to existing files, or
in-place cell edits to existing notebooks.

---

## Architecture

The pipeline is a linear sequence of Jupyter notebooks backed by a small set of
Python source modules. Bug fixes do not change this linear flow; they harden each
phase against errors that would propagate into downstream phases.

```
raw data
  │
  ▼
Phase 1  01_data_quality_and_rul.ipynb          [+SEED]
  │  train_with_rul.csv
  ▼
Phase 2  02_preprocessing.ipynb                 [+SEED]
  │  src/preprocessing/preprocess.py            [+save scalers, +subset param,
  │                                              +normalize_per_engine_apply,
  │                                              +preprocess_test_set]
  │  X_train.npy  y_train.npy  scalers.pkl
  ▼
Phase 3  03_physics_labels.ipynb                [+SEED, +mkdir reports/figures]
  │  labels_train.npy
  ▼
Phase 4  04_model_architecture.ipynb            [+SEED, +mkdir reports/figures]
  ▼
Phase 5  05_training.ipynb                      [+SEED, +mkdir reports/figures]
  ▼
Phase 5b 05b_training_cgan.ipynb                [+SEED, +mkdir reports/figures,
  │                                              +early stopping w/ disc score]
  │  checkpoints_cgan/{generator,discriminator,model_config.json}.pt
  ▼
Phase 6  06_synthetic_fault_generation.ipynb    [MODELS_DIR → checkpoints_cgan,
  │                                              +CGAN arch, +key inspection,
  │  synth_X.npy  synth_labels.npy               +dead-sensor rescaling,
  │  interp_C1_C2.npy  interp_C1_C2_labels.npy   +interp labels,
  │  interp_C2_C3.npy  interp_C2_C3_labels.npy   +mkdir reports/figures]
  ▼
Phase 7  07_validation.ipynb                    [+SEED, load from Synth_Dir,
  │                                              FileNotFoundError guard,
  │                                              +mkdir reports/figures]
  ▼
Phase 8  (future) 08_classifier.ipynb
         preprocess_test_set() → X_test.npy  y_test_rul.npy
```

---

## Components and Interfaces

### A. `src/preprocessing/preprocess.py` — additions

#### A.1  `normalize_per_engine` (modified)

Current signature: `normalize_per_engine(df) -> pd.DataFrame`

New signature and logic:

```python
def normalize_per_engine(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, MinMaxScaler]]:
    """
    Fit one MinMaxScaler per engine on FEATURE_COLS.
    Returns:
      - scaled DataFrame (same shape as input)
      - scalers dict {engine_id: fitted_scaler}
    """
    df = df.copy()
    scalers: dict[int, MinMaxScaler] = {}
    scaled_parts = []
    for engine_id, group in df.groupby("engine_id"):
        scaler = MinMaxScaler()
        group = group.copy()
        group[FEATURE_COLS] = scaler.fit_transform(group[FEATURE_COLS])
        scalers[int(engine_id)] = scaler
        scaled_parts.append(group)
    scaled_df = pd.concat(scaled_parts).sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)
    return scaled_df, scalers
```

The existing call site inside `run_preprocessing` is updated to unpack both
return values and pass `scalers` to `joblib.dump`.

**Rationale:** Returning the dict from the function (rather than a side-effect
inside it) keeps the function pure and testable.

#### A.2  `normalize_per_engine_apply` (new)

```python
def normalize_per_engine_apply(
    df: pd.DataFrame,
    scalers_path: Path,
) -> pd.DataFrame:
    """
    Transform FEATURE_COLS using pre-fitted scalers loaded from scalers_path.
    Raises FileNotFoundError if scalers_path does not exist.
    Raises KeyError("{engine_id}") if an engine in df has no entry in the dict.
    """
    scalers_path = Path(scalers_path)
    if not scalers_path.exists():
        raise FileNotFoundError(
            f"Scalers file not found: {scalers_path}"
        )
    scalers: dict[int, MinMaxScaler] = joblib.load(scalers_path)
    df = df.copy()
    parts = []
    for engine_id, group in df.groupby("engine_id"):
        eid = int(engine_id)
        if eid not in scalers:
            raise KeyError(
                f"Engine ID {eid} not found in saved scalers dict "
                f"(path: {scalers_path})"
            )
        group = group.copy()
        group[FEATURE_COLS] = scalers[eid].transform(group[FEATURE_COLS])
        parts.append(group)
    return pd.concat(parts).sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)
```

#### A.3  `preprocess_test_set` (new)

```python
COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}"  for i in range(1, 22)]
)

def preprocess_test_set(
    test_path: Path,
    rul_path: Path,
    scalers_path: Path,
    output_dir: Path,
    window_size: int = 30,
) -> None:
    """
    Preprocess test_FD001.txt using saved training scalers.
    For each engine, retain only the LAST complete window of size window_size.
    Engines with fewer than window_size cycles are skipped with a warning.

    Outputs:
      output_dir/X_test.npy     shape (N_valid, 30, 17)  float32
      output_dir/y_test_rul.npy shape (N_valid,)          float32
    """
    test_path     = Path(test_path)
    rul_path      = Path(rul_path)
    scalers_path  = Path(scalers_path)
    output_dir    = Path(output_dir)

    if not scalers_path.exists():
        raise FileNotFoundError(
            f"Scalers file not found: {scalers_path}"
        )

    # Load raw test data
    df = pd.read_csv(test_path, sep=r"\s+", header=None,
                     names=COLUMN_NAMES)
    df = df.drop(columns=DROP_SENSORS)

    # Apply saved scalers (no refit)
    df = normalize_per_engine_apply(df, scalers_path)

    # Load ground-truth RUL (one value per engine, ordered by engine_id)
    rul_values = pd.read_csv(rul_path, header=None, names=["RUL"])[
        "RUL"
    ].values.astype(np.float32)

    X_windows = []
    valid_engine_indices = []   # indices into rul_values (0-based)

    for idx, (engine_id, group) in enumerate(
        df.groupby("engine_id", sort=True)
    ):
        group = group.sort_values("cycle")
        features = group[FEATURE_COLS].values
        n_cycles = len(features)

        if n_cycles < window_size:
            import warnings
            warnings.warn(
                f"Engine {engine_id} has only {n_cycles} cycles "
                f"(< window_size={window_size}); skipping.",
                stacklevel=2,
            )
            continue

        # last complete window
        last_window = features[-window_size:]          # (30, 17)
        X_windows.append(last_window)
        valid_engine_indices.append(idx)

    X_test = np.array(X_windows, dtype=np.float32)    # (N_valid, 30, 17)
    y_test  = rul_values[valid_engine_indices]          # (N_valid,)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_test.npy",     X_test)
    np.save(output_dir / "y_test_rul.npy", y_test)

    print(f"X_test shape    : {X_test.shape}")
    print(f"y_test_rul shape: {y_test.shape}")
    print(f"Saved to        : {output_dir.resolve()}")
```

#### A.4  `run_preprocessing` (modified)

```python
VALID_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}

def run_preprocessing(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    window_size: int = 30,
    stride: int = 1,
    subset: str = "FD001",
) -> None:
    if subset not in VALID_SUBSETS:
        raise ValueError(
            f"Invalid subset {subset!r}. "
            f"Valid values: {sorted(VALID_SUBSETS)}"
        )

    if input_path is None:
        input_path = Path(f"data/processed/{subset}/train_with_rul.csv")
    if output_dir is None:
        output_dir = Path(f"data/processed/{subset}")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {input_path}"
        )

    df = pd.read_csv(input_path)
    df = cap_rul(df)
    df, scalers = normalize_per_engine(df)        # <-- unpacks both values

    X, y = make_windows(df, window_size=window_size, stride=stride)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X)
    np.save(output_dir / "y_train.npy", y)
    joblib.dump(scalers, output_dir / "scalers.pkl")  # <-- save scalers

    print(f"X_train shape : {X.shape}")
    print(f"y_train shape : {y.shape}")
    print(f"Scalers saved : {(output_dir / 'scalers.pkl').resolve()}")
    print(f"Saved to      : {output_dir.resolve()}")
```

---

### B. Notebook Cell Edits

#### B.1  Reproducibility seeds — all notebooks (BUG-005)

Insert at the very top of **Cell 1** in each of the 8 notebooks listed in
Requirement 4. The exact block varies by notebook:

**Notebooks without PyTorch** (01, 02, 03):
```python
SEED = 42
import numpy as np
np.random.seed(SEED)
```

**Notebooks with PyTorch** (04, 05, 05b, 06, 07):
```python
SEED = 42
import numpy as np
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

For any cell in a notebook that calls `RandomForestClassifier` or
`train_test_split`, add `random_state=SEED` to that call.

#### B.2  `reports/figures/` auto-creation — notebooks 03–07 and 05b (BUG-012)

Append to **Cell 1** of each affected notebook (03, 04, 05, 05b, 06, 07):

```python
from pathlib import Path
Path("../reports/figures").mkdir(parents=True, exist_ok=True)
```

If `Path` is already imported in Cell 1, only the `mkdir` call is added.

#### B.3  Phase 6 — point to CGAN checkpoints (BUG-002)

In `06_synthetic_fault_generation.ipynb` **Cell 1**, change:

```python
# BEFORE
MODELS_DIR = Path("../data/processed/FD001/checkpoints")
# ...
with open("../configs/model_config.json") as f:
    cfg = json.load(f)
```

```python
# AFTER
MODELS_DIR = Path("../data/processed/FD001/checkpoints_cgan")
# ...
with open(MODELS_DIR / "model_config.json") as f:
    cfg = json.load(f)
```

The second spurious cell that also sets `MODELS_DIR = Path(...checkpoints)` is
deleted entirely (it was a diagnostic key-inspection cell and must not override
the corrected Cell 1 value).

#### B.4  Phase 6 — replace TimeGAN model classes with CGAN classes (BUG-002)

Remove the cell that defines `Embedder`, `Recovery`, `Supervisor`, `Generator`,
`Discriminator` (TimeGAN classes) and their `.load_state_dict` calls.

Replace with a new cell that defines `CGANGenerator` and `CGANDiscriminator`
(copy verbatim from `05b_training_cgan.ipynb` Cells 3–4) and loads:

```python
G_net = CGANGenerator(LATENT_DIM, HIDDEN_DIM, SEQ_LEN,
                      INPUT_DIM, NUM_CLASSES, EMBED_DIM).to(DEVICE)
D_net = CGANDiscriminator(INPUT_DIM, HIDDEN_DIM,
                          NUM_CLASSES, EMBED_DIM).to(DEVICE)

def _safe_load(model, ckpt_path):
    """Load state dict with key validation (BUG-007)."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    expected = set(model.state_dict().keys())
    actual   = set(ckpt.keys())
    if expected != actual:
        raise ValueError(
            f"Checkpoint key mismatch for {ckpt_path.name}:\n"
            f"  Expected keys : {sorted(expected)}\n"
            f"  Checkpoint keys: {sorted(actual)}"
        )
    model.load_state_dict(ckpt, strict=True)

_safe_load(G_net, MODELS_DIR / "generator.pt")
_safe_load(D_net, MODELS_DIR / "discriminator.pt")
G_net.eval(); D_net.eval()
print("CGAN checkpoints loaded.")
```

This implements BUG-007 (safe loading) inline.

#### B.5  Phase 6 — update `generate_samples` to use CGAN (BUG-002)

Replace the TimeGAN generation function that calls
`generator → supervisor → recovery` with the CGAN version:

```python
def generate_samples(generator, n_samples, class_id, device):
    """
    Generate n_samples synthetic windows for a given class_id using CGAN.
    Returns numpy array of shape (n_samples, SEQ_LEN, INPUT_DIM).
    """
    all_samples = []
    batch_size  = 128
    with torch.no_grad():
        generated = 0
        while generated < n_samples:
            bs = min(batch_size, n_samples - generated)
            z  = torch.randn(bs, LATENT_DIM).to(device)
            c  = torch.full((bs,), class_id, dtype=torch.long).to(device)
            x_hat = generator(z, c).cpu().numpy()       # (bs, T, D)
            all_samples.append(x_hat)
            generated += bs
    return np.concatenate(all_samples, axis=0)
```

All downstream calls to `generate_samples` drop the `supervisor` and `recovery`
arguments and pass `G_net` instead.

#### B.6  Phase 6 — dead-sensor rescaling (BUG-004)

Add a new cell immediately after the generation loop (before saving):

```python
# ── Post-process: rescale synthetic features to match real statistics ──
X_real_flat = X.reshape(-1, INPUT_DIM)          # (N*30, 17)
real_mean    = X_real_flat.mean(axis=0)          # (17,)
real_std     = X_real_flat.std(axis=0)           # (17,)
EPS = 1e-8

def rescale_to_real_stats(synth: np.ndarray) -> np.ndarray:
    """
    synth: (N, 30, 17) float32
    Returns rescaled array clipped to [0, 1].
    """
    flat = synth.reshape(-1, INPUT_DIM)           # (N*30, 17)
    raw_mean = flat.mean(axis=0)
    raw_std  = flat.std(axis=0)
    flat_rescaled = (
        (flat - raw_mean) / (raw_std + EPS)
    ) * real_std + real_mean
    flat_rescaled = np.clip(flat_rescaled, 0.0, 1.0)
    return flat_rescaled.reshape(synth.shape).astype(np.float32)

synth_X_rescaled = rescale_to_real_stats(synth_X)
```

Then save `synth_X_rescaled` as `synth_X.npy` instead of the raw `synth_X`.

#### B.7  Phase 6 — save interpolation labels (BUG-006)

In the interpolation cell (previously Cell 8), immediately after saving each
interpolation array, add label saves:

```python
# interp C1 <-> C2
interp_C1_C2 = ...                                # existing interpolation code
np.save(SYNTH_DIR / "interp_C1_C2.npy",        interp_C1_C2)
np.save(SYNTH_DIR / "interp_C1_C2_labels.npy",
        np.ones(len(interp_C1_C2), dtype=np.int64))

# interp C2 <-> C3
interp_C2_C3 = ...                                # existing interpolation code
np.save(SYNTH_DIR / "interp_C2_C3.npy",        interp_C2_C3)
np.save(SYNTH_DIR / "interp_C2_C3_labels.npy",
        np.full(len(interp_C2_C3), 2, dtype=np.int64))
```

#### B.8  Phase 7 — load CGAN data and guard against missing files (BUG-008)

In `07_validation.ipynb` **Cell 1**, replace any hard-coded path references with:

```python
SYNTH_DIR = Path("../data/synthetic/GAN")

synth_X_path     = SYNTH_DIR / "synth_X.npy"
synth_labels_path = SYNTH_DIR / "synth_labels.npy"

if not synth_X_path.exists() or not synth_labels_path.exists():
    raise FileNotFoundError(
        "CGAN synthetic data not found. "
        "Please run Phase 6 (06_synthetic_fault_generation.ipynb) first.\n"
        f"  Missing: {synth_X_path}\n"
        f"  Missing: {synth_labels_path}"
    )

X_synth = np.load(synth_X_path)
y_synth = np.load(synth_labels_path)
```

All subsequent cells that previously referenced TimeGAN arrays or
`checkpoints/` paths are updated to use `X_synth` and `y_synth`.

#### B.9  Phase 5b — early stopping (BUG-013)

Replace the training loop in **Cell 6** of `05b_training_cgan.ipynb` with the
version below. The loop is otherwise identical to the existing code; only the
early-stopping block and the final checkpoint-save are added.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler as SKStdScaler

EARLY_STOP_THRESHOLD = 0.60
EVAL_INTERVAL        = 50

def compute_discriminative_score(G_model, X_real, y_real, n_samples=500,
                                  seed=42):
    """
    Draw n_samples real and n_samples synthetic windows.
    Train a RandomForest in 5-fold CV to distinguish them.
    Returns mean accuracy (ideal ~0.50).
    """
    # Sample real windows stratified by class
    rng         = np.random.default_rng(seed)
    indices     = rng.choice(len(X_real), size=n_samples, replace=False)
    X_real_samp = X_real[indices].reshape(n_samples, -1)   # flatten time dim
    y_real_cls  = y_real[indices]

    # Generate synthetic
    with torch.no_grad():
        z  = torch.randn(n_samples, LATENT_DIM).to(DEVICE)
        c  = torch.tensor(y_real_cls, dtype=torch.long).to(DEVICE)
        X_fake_t = G_model(z, c).cpu().numpy()
    X_fake_samp = X_fake_t.reshape(n_samples, -1)

    # Build binary classification dataset (0=real, 1=fake)
    X_all  = np.vstack([X_real_samp, X_fake_samp])
    y_disc = np.array([0]*n_samples + [1]*n_samples)

    sc = SKStdScaler()
    X_all = sc.fit_transform(X_all)

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    scores = cross_val_score(clf, X_all, y_disc, cv=5)
    return scores.mean()


print(f"Training CGAN for {EPOCHS} epochs...")
early_stopped = False

for epoch in range(EPOCHS):
    g_epoch, d_epoch = 0.0, 0.0

    for x_real, c_batch in dataloader:
        # ... existing D and G update code (unchanged) ...
        pass

    g_avg = g_epoch / len(dataloader)
    d_avg = d_epoch / len(dataloader)
    history_g.append(g_avg)
    history_d.append(d_avg)

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:>4}/{EPOCHS}  G={g_avg:.4f}  D={d_avg:.4f}")

    # Early stopping evaluation at every 50th epoch
    if (epoch + 1) % EVAL_INTERVAL == 0:
        disc_score = compute_discriminative_score(
            G, X, labels, n_samples=500, seed=SEED
        )
        print(f"    Discriminative score @ epoch {epoch+1}: {disc_score:.4f}")

        if disc_score < EARLY_STOP_THRESHOLD:
            print(
                f"Early stopping triggered at epoch {epoch+1} "
                f"(discriminative score {disc_score:.4f} < "
                f"{EARLY_STOP_THRESHOLD})."
            )
            early_stopped = True
            break

if not early_stopped:
    print("Training complete (full 500 epochs).")

# Save checkpoints regardless of how training ended
torch.save(G.state_dict(), MODELS_DIR / "generator.pt")
torch.save(D.state_dict(), MODELS_DIR / "discriminator.pt")
with open(MODELS_DIR / "model_config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Checkpoints saved to {MODELS_DIR.resolve()}")
```

---

## Data Models

### Scalers file

| Key         | Type                          | Notes                            |
|-------------|-------------------------------|----------------------------------|
| engine_id   | `int`                         | Matches `engine_id` column value |
| scaler      | `sklearn.preprocessing.MinMaxScaler` | Fitted on FEATURE_COLS    |

Serialized with `joblib.dump` to `data/processed/{subset}/scalers.pkl`.

### Interpolation label arrays

| File                         | Shape         | dtype   | Value |
|------------------------------|---------------|---------|-------|
| `interp_C1_C2_labels.npy`    | `(500,)`      | `int64` | `1`   |
| `interp_C2_C3_labels.npy`    | `(500,)`      | `int64` | `2`   |

### Test arrays (Phase 8 input)

| File             | Shape                   | dtype     | Description                        |
|------------------|-------------------------|-----------|------------------------------------|
| `X_test.npy`     | `(N_valid, 30, 17)`     | `float32` | Last window per valid test engine  |
| `y_test_rul.npy` | `(N_valid,)`            | `float32` | Ground-truth RUL from RUL_FD001.txt|

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid
executions of a system — essentially, a formal statement about what the system
should do. Properties serve as the bridge between human-readable specifications
and machine-verifiable correctness guarantees.*

### Property 1: Scalers_Dict completeness

*For any* training DataFrame containing `k` distinct engine IDs, calling
`normalize_per_engine` SHALL return a `Scalers_Dict` whose key set is exactly
equal to the set of integer engine IDs present in the DataFrame.

**Validates: Requirements 1.1**

---

### Property 2: Scaler apply — values in unit range

*For any* training DataFrame, if `normalize_per_engine` is called to produce a
`Scalers_Dict` and `normalize_per_engine_apply` is then called on the same
DataFrame using the saved scalers, all values in `FEATURE_COLS` of the output
SHALL lie in the closed interval `[0.0, 1.0]` (to within floating-point
precision).

**Validates: Requirements 1.3**

---

### Property 3: Rescaling formula correctness

*For any* synthetic array of shape `(N, 30, 17)` and any target per-feature
mean and standard deviation, after applying the rescaling formula
`x_rescaled = (x_raw − raw_mean) / (raw_std + eps) * real_std + real_mean`
followed by clipping to `[0, 1]`:
- Every value in the output SHALL be in `[0.0, 1.0]`.
- The absolute difference between the output's per-feature mean and the target
  `real_mean` SHALL be less than 0.05 (over any array with N ≥ 100 samples,
  when no clipping occurs at the mean).

**Validates: Requirements 3.2, 3.4**

---

### Property 4: Deterministic pipeline functions

*For any* training DataFrame, calling `normalize_per_engine` twice in
independent invocations after seeding `numpy.random` with `SEED = 42` SHALL
produce identical output DataFrames (bit-for-bit equal float values in all
`FEATURE_COLS`). Similarly, `make_windows` given the same DataFrame and the
same parameters SHALL return identical `X` and `y` arrays across runs.

**Validates: Requirements 4.5**

---

### Property 5: Test windowing correctness and output alignment

*For any* test DataFrame in which each engine has a variable number of cycles
≥ 30, calling the last-window extraction logic SHALL produce:
- An `X_test` array of shape `(N_valid, 30, 17)` where `N_valid` equals the
  count of engines with ≥ 30 cycles.
- A `y_test_rul` array of shape `(N_valid,)`.
- `X_test[i]` exactly matching the last 30 rows of `FEATURE_COLS` for the
  `i`-th valid engine.

*For any* engine with fewer than 30 cycles, that engine SHALL be absent from
`X_test` and `y_test_rul`.

**Validates: Requirements 8.4, 8.5, 8.6**

---

### Property 6: Subset-routing correctness

*For any* valid subset name `s` in `{"FD001", "FD002", "FD003", "FD004"}`,
calling `run_preprocessing(subset=s)` SHALL write `X_train.npy`, `y_train.npy`,
and `scalers.pkl` exclusively to `data/processed/{s}/`, and SHALL NOT write any
output to any other subset's directory.

**Validates: Requirements 9.1, 9.2, 9.3**

---

### Property 7: Early stopping control flow

*For any* sequence of discriminative scores evaluated at epochs
50, 100, 150, …, 500, the training loop SHALL exit at the epoch corresponding
to the first score that is strictly less than 0.60. If no score in the sequence
is less than 0.60, the loop SHALL complete all 500 epochs. In both cases,
checkpoints SHALL be saved to `CGAN_Checkpoints_Dir` after exit.

**Validates: Requirements 11.2, 11.3, 11.4, 11.5**

---

## Error Handling

| Scenario | Location | Behaviour |
|---|---|---|
| `scalers.pkl` missing when `normalize_per_engine_apply` called | `preprocess.py` | `FileNotFoundError` with file path |
| Engine ID in test data has no scaler entry | `preprocess.py` | `KeyError` with engine ID |
| `scalers.pkl` missing when `preprocess_test_set` called | `preprocess.py` | `FileNotFoundError` with file path |
| Test engine has < 30 cycles | `preprocess.py` | `warnings.warn` + engine skipped |
| Invalid subset name in `run_preprocessing` | `preprocess.py` | `ValueError` listing valid names |
| `train_with_rul.csv` missing for subset | `preprocess.py` | `FileNotFoundError` with path |
| Checkpoint key mismatch in Phase 6 `_safe_load` | notebook cell | `ValueError` with expected vs actual key sets |
| `synth_X.npy` or `synth_labels.npy` missing in Phase 7 | notebook Cell 1 | `FileNotFoundError` instructing user to run Phase 6 |
| `reports/figures/` creation fails (permission error) | notebook Cell 1 | OS error propagated immediately |

---

## Testing Strategy

### Overview

PBT is appropriate for this feature because several of the fixed functions are
pure data transformations (scaler fitting, test windowing, rescaling,
early-stopping control flow) where input variation reveals correctness
properties. We use **Hypothesis** (Python) as the PBT library.

Unit tests cover specific examples, edge cases, and error conditions.
Integration tests cover notebook-level outcomes (file existence, path
correctness).

All property tests are configured to run a minimum of 100 examples each.

### Unit tests — `tests/test_preprocess.py`

| Test | Type | What it verifies |
|---|---|---|
| `test_scalers_dict_keys` | Property (P1) | Scalers_Dict keys == input engine IDs |
| `test_apply_values_in_range` | Property (P2) | All FEATURE_COLS in [0,1] after apply |
| `test_rescaling_clipping` | Property (P3) | Rescaled values in [0,1] |
| `test_rescaling_mean_match` | Property (P3) | Mean difference < 0.05 |
| `test_determinism_normalize` | Property (P4) | normalize_per_engine deterministic |
| `test_determinism_make_windows` | Property (P4) | make_windows deterministic |
| `test_last_window_correctness` | Property (P5) | X_test rows match last 30 cycles |
| `test_output_shape_alignment` | Property (P5) | X_test.shape[0] == y_test_rul.shape[0] |
| `test_short_engine_skipped` | Property (P5) | Engines < 30 cycles absent from output |
| `test_subset_routing` | Property (P6) | Correct output dir for each subset |
| `test_invalid_subset_raises` | Edge case | ValueError on bad subset name |
| `test_missing_scalers_raises` | Edge case | FileNotFoundError on missing pkl |
| `test_missing_engine_id_raises` | Edge case | KeyError on unknown engine ID |
| `test_missing_train_csv_raises` | Edge case | FileNotFoundError on missing CSV |
| `test_scalers_serialized` | Example | scalers.pkl written and loadable |
| `test_interp_labels_shape_dtype` | Example | label arrays are int64 with correct values |

### Unit tests — `tests/test_early_stopping.py`

| Test | Type | What it verifies |
|---|---|---|
| `test_early_stop_fires_at_first_breach` | Property (P7) | Loop exits at first score < 0.60 |
| `test_no_early_stop_runs_all_epochs` | Property (P7) | Full 500 epochs when no breach |
| `test_checkpoints_saved_after_early_stop` | Example | Saves at early-stop exit |
| `test_checkpoints_saved_after_full_run` | Example | Saves after full 500 epochs |

### Property test configuration

```python
# pytest + hypothesis setup
from hypothesis import given, settings
settings.register_profile("ci", max_examples=100)
settings.load_profile("ci")
```

Each property test is tagged in a comment:
```python
# Feature: cmapss-bug-fixes, Property 1: Scalers_Dict completeness
```

### Integration tests

- Verify `06_synthetic_fault_generation.ipynb` Cell 1 contains `checkpoints_cgan`
  path (static notebook JSON parse — no execution required).
- Verify `reports/figures/` is created by running the Cell 1 setup snippet in a
  temp directory.
- After a full Phase 6 run: verify per-feature mean differences < 0.05 between
  `synth_X.npy` and `X_train.npy`.

### What is not property-tested

- Notebook-level path configuration checks (static analysis only — no input
  variation makes PBT wasteful).
- `reports/figures/` directory creation (side-effect only; example test).
- Phase 7 metrics (integration test with real generated data).
- CGAN training convergence (not a unit-testable property; verified by
  discriminative score monitoring during training).
