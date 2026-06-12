import numpy as np
import pandas as pd

# BUG-005 fix: reproducibility seed
SEED = 42
np.random.seed(SEED)

MONOTONE_INCREASING = ["s4"]  
MONOTONE_DECREASING = ["s3", "s7", "s11"] 

FEATURE_COLS = [
    "op1", "op2", "op3",
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]


def check_monotonicity(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    For each physics-constrained sensor, compute per-class mean
    and check monotone direction holds C0->C1->C2->C3.

    X shape: (N, 30, 14)
    labels:  (N,)
    """

    X_mean = X.mean(axis=1)

    results = []
    for sensor in MONOTONE_INCREASING + MONOTONE_DECREASING:
        if sensor not in FEATURE_COLS:
            continue
        idx = FEATURE_COLS.index(sensor)
        direction = "increase" if sensor in MONOTONE_INCREASING else "decrease"
        class_means = []
        for c in range(4):
            mask = labels == c
            if mask.sum() == 0:
                class_means.append(np.nan)
            else:
                class_means.append(X_mean[mask, idx].mean())

        means = [m for m in class_means if not np.isnan(m)]
        if direction == "increase":
            holds = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
        else:
            holds = all(means[i] >= means[i + 1] for i in range(len(means) - 1))

        results.append({
            "sensor": sensor,
            "direction": direction,
            "C0_mean": class_means[0],
            "C1_mean": class_means[1],
            "C2_mean": class_means[2],
            "C3_mean": class_means[3],
            "monotone_holds": holds,
        })

    return pd.DataFrame(results)