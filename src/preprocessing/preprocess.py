import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

# BUG-005 fix: reproducibility seed
SEED = 42
np.random.seed(SEED)

RUL_CAP = 125

DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

SENSOR_COLS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP_SENSORS]
OP_COLS = ["op1", "op2", "op3"]
FEATURE_COLS = OP_COLS + SENSOR_COLS


def cap_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def normalize_per_engine(
    df: pd.DataFrame,
    scalers: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Normalize features per engine.

    If fit=True (training), fit new scalers and return them.
    If fit=False (test), apply pre-fitted scalers from `scalers` dict.
    """
    df = df.copy()
    if scalers is None:
        scalers = {}
    scaled_parts = []
    for engine_id, group in df.groupby("engine_id"):
        group = group.copy()
        if fit:
            scaler = MinMaxScaler()
            group[FEATURE_COLS] = scaler.fit_transform(group[FEATURE_COLS])
            scalers[engine_id] = scaler
        else:
            if engine_id in scalers:
                scaler = scalers[engine_id]
                group[FEATURE_COLS] = scaler.transform(group[FEATURE_COLS])
            else:
                # BUG-015 fix: do NOT refit on unseen engine data (information
                # leakage). Raise so callers apply the production scaler or
                # explicitly decide how to handle an unknown engine.
                raise KeyError(
                    f"Engine '{engine_id}' was not seen during training. "
                    "Pass the saved training scalers and ensure all test "
                    "engines are covered, or use the global_scaler for "
                    "inference (see turbofan_app/backend/app.py)."
                )
        scaled_parts.append(group)
    result = pd.concat(scaled_parts).sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    return result, scalers


def make_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        features = group[FEATURE_COLS].values
        labels = group["RUL"].values
        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            X.append(features[start:end])
            y.append(labels[end - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_preprocessing(
    input_path: Path,
    output_dir: Path,
    window_size: int = 30,
    stride: int = 1,
) -> None:
    df = pd.read_csv(input_path)

    df = cap_rul(df)
    df, scalers = normalize_per_engine(df, fit=True)

    # BUG-001 fix: save scalers so test set can reuse them without data leakage
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, output_dir / "scalers.pkl")
    print(f"Saved scalers -> {output_dir / 'scalers.pkl'}")

    X, y = make_windows(df, window_size=window_size, stride=stride)

    np.save(output_dir / "X_train.npy", X)
    np.save(output_dir / "y_train.npy", y)

    print(f"X_train shape : {X.shape}")   # (N, 30, 17)
    print(f"y_train shape : {y.shape}")   # (N,)
    print(f"Saved to      : {output_dir.resolve()}")


if __name__ == "__main__":
    run_preprocessing(
        input_path=Path("data/processed/FD001/train_with_rul.csv"),
        output_dir=Path("data/processed/FD001"),
    )