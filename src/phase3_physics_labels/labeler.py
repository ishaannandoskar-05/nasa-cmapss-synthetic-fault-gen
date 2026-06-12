import numpy as np
import pandas as pd
from pathlib import Path

# BUG-005 fix: reproducibility seed
SEED = 42
np.random.seed(SEED)

FAULT_CLASSES = {
    "C0": (100, float("inf")),
    "C1": (50, 100),
    "C2": (10, 50),
    "C3": (0, 10),
}

CLASS_LABELS = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}


def assign_fault_class(rul: float) -> int:
    if rul > 100:
        return CLASS_LABELS["C0"]
    elif rul > 50:
        return CLASS_LABELS["C1"]
    elif rul > 10:
        return CLASS_LABELS["C2"]
    else:
        return CLASS_LABELS["C3"]


def label_windows(y: np.ndarray) -> np.ndarray:
    vectorized = np.vectorize(assign_fault_class)
    return vectorized(y).astype(np.int64)


def class_distribution(labels: np.ndarray) -> pd.DataFrame:
    unique, counts = np.unique(labels, return_counts=True)
    class_names = {v: k for k, v in CLASS_LABELS.items()}
    df = pd.DataFrame({
        "class_id": unique,
        "class_name": [class_names[i] for i in unique],
        "count": counts,
        "pct": counts / counts.sum() * 100,
    })
    return df


def run_labeling(processed_dir: Path) -> None:
    y = np.load(processed_dir / "y_train.npy")
    labels = label_windows(y)
    np.save(processed_dir / "labels_train.npy", labels)
    dist = class_distribution(labels)
    print(dist.to_string(index=False))
    print(f"\nSaved labels_train.npy -> {processed_dir.resolve()}")



if __name__ == "__main__":
    run_labeling(Path("data/processed/FD001"))