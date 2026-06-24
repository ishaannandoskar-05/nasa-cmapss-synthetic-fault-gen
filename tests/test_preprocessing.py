"""
Automated pytest suite for CMAPSS preprocessing and scaler validation.
Auto-generated from tests/generate_tests.ipynb - do not edit results by hand,
rerun the notebook to regenerate this file if the pipeline changes.

Run with: pytest tests/test_preprocessing.py -v
"""

import json
import numpy as np
import pandas as pd
import joblib
import pytest
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

FEATURE_COLS = [
    "op1","op2",
    "s2","s3","s4","s7","s8","s9",
    "s11","s12","s13","s14","s15","s17","s20","s21",
]
GLOBAL_SCALE_FEATURES = ["s3"]
PER_ENGINE_FEATURES   = [f for f in FEATURE_COLS if f not in GLOBAL_SCALE_FEATURES]

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)
DROP_COLS = ["s1","s5","s6","s10","s16","s18","s19","op3"]
SEQ_LEN = 30


def load_cmapss(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :len(COLUMNS)]
    df.columns = COLUMNS
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def rul_to_class(rul):
    if rul > 100: return 0
    elif rul > 50: return 1
    elif rul > 10: return 2
    else: return 3


def apply_saved_scalers(df, global_scaler, engine_scalers):
    df = df.copy()
    df[GLOBAL_SCALE_FEATURES] = global_scaler.transform(df[GLOBAL_SCALE_FEATURES])
    scaled_parts = []
    for engine_id, group in df.groupby("engine_id"):
        group = group.copy()
        if engine_id in engine_scalers:
            group[PER_ENGINE_FEATURES] = engine_scalers[engine_id].transform(group[PER_ENGINE_FEATURES])
        else:
            group[PER_ENGINE_FEATURES] = MinMaxScaler().fit_transform(group[PER_ENGINE_FEATURES])
        scaled_parts.append(group)
    return pd.concat(scaled_parts).sort_values(["engine_id","cycle"]).reset_index(drop=True)


def get_last_window(df, window_size=SEQ_LEN):
    X = []
    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        feats = group[FEATURE_COLS].values
        if len(feats) >= window_size:
            window = feats[-window_size:]
        else:
            pad = np.zeros((window_size - len(feats), feats.shape[1]))
            window = np.vstack([pad, feats])
        X.append(window)
    return np.array(X, dtype=np.float32)


@pytest.fixture(params=SUBSETS)
def subset(request):
    return request.param


@pytest.fixture
def test_raw(subset):
    path = RAW_DIR / f"test_{subset}.txt"
    if not path.exists():
        pytest.skip(f"test_{subset}.txt not found")
    return load_cmapss(path)


@pytest.fixture
def rul_df(subset):
    path = RAW_DIR / f"RUL_{subset}.txt"
    if not path.exists():
        pytest.skip(f"RUL_{subset}.txt not found")
    return pd.read_csv(path, header=None, names=["RUL"])


@pytest.fixture
def scalers(subset):
    gs_path = PROCESSED_DIR / subset / "global_scaler.pkl"
    es_path = PROCESSED_DIR / subset / "engine_scalers.pkl"
    if not (gs_path.exists() and es_path.exists()):
        pytest.skip(f"scalers for {subset} not found")
    return joblib.load(gs_path), joblib.load(es_path)


class TestRawDataIntegrity:
    def test_no_nans(self, test_raw):
        assert test_raw.isnull().sum().sum() == 0

    def test_has_feature_columns(self, test_raw):
        missing = [c for c in FEATURE_COLS if c not in test_raw.columns]
        assert not missing, f"missing columns: {missing}"

    def test_op3_dropped(self, test_raw):
        assert "op3" not in test_raw.columns

    def test_rul_count_matches_engines(self, test_raw, rul_df):
        assert len(rul_df) == test_raw["engine_id"].nunique()


class TestScalerApplication:
    def test_global_scaler_is_minmax(self, scalers):
        global_scaler, _ = scalers
        assert isinstance(global_scaler, MinMaxScaler)

    def test_engine_scalers_nonempty(self, scalers):
        _, engine_scalers = scalers
        assert isinstance(engine_scalers, dict) and len(engine_scalers) > 0

    def test_scaled_values_in_range(self, test_raw, scalers):
        global_scaler, engine_scalers = scalers
        scaled = apply_saved_scalers(test_raw, global_scaler, engine_scalers)
        assert scaled[FEATURE_COLS].min().min() >= -0.05
        assert scaled[FEATURE_COLS].max().max() <= 1.05

    def test_no_row_loss_after_scaling(self, test_raw, scalers):
        global_scaler, engine_scalers = scalers
        scaled = apply_saved_scalers(test_raw, global_scaler, engine_scalers)
        assert len(scaled) == len(test_raw)

    def test_s3_has_variance_after_global_scaling(self, test_raw, scalers):
        global_scaler, engine_scalers = scalers
        scaled = apply_saved_scalers(test_raw, global_scaler, engine_scalers)
        assert scaled["s3"].std() > 0.001


class TestWindowConstruction:
    def test_window_shape(self, test_raw, scalers):
        global_scaler, engine_scalers = scalers
        scaled = apply_saved_scalers(test_raw, global_scaler, engine_scalers)
        X = get_last_window(scaled)
        n_engines = test_raw["engine_id"].nunique()
        assert X.shape == (n_engines, SEQ_LEN, len(FEATURE_COLS))

    def test_no_nans_in_windows(self, test_raw, scalers):
        global_scaler, engine_scalers = scalers
        scaled = apply_saved_scalers(test_raw, global_scaler, engine_scalers)
        X = get_last_window(scaled)
        assert not np.isnan(X).any()


class TestLabelConsistency:
    def test_rul_to_class_boundaries(self, rul_df):
        for rul in rul_df["RUL"].values:
            c = rul_to_class(rul)
            assert c in {0, 1, 2, 3}
            if c == 0: assert rul > 100
            if c == 1: assert 50 < rul <= 100
            if c == 2: assert 10 < rul <= 50
            if c == 3: assert rul <= 10
