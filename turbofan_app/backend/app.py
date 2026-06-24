"""
Flask API serving the unified 1D-CNN turbofan fault classifier.

Endpoints:
  GET  /api/health           -> service status
  POST /api/predict          -> upload CSV window, get fault class + RUL
  GET  /api/classes          -> class definitions and fault zone mapping
  GET  /api/sample/<class_id>-> synthetic sample window for demo purposes

Run:
  pip install flask flask-cors torch numpy pandas scikit-learn joblib
  python app.py
"""

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model_artifacts"

FEATURE_COLS = [
    "op1", "op2",
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
SEQ_LEN = 30
INPUT_DIM = 16
NUM_CLASSES = 4

CLASS_NAMES = ["C0 healthy", "C1 early", "C2 advanced", "C3 imminent"]
CLASS_RUL_MIDPOINT = {0: 125, 1: 75, 2: 30, 3: 5}

# fault zone mapping consumed by the frontend 3D viewer
FAULT_ZONE_MAP = {
    0: [],
    1: ["hpc"],
    2: ["hpc", "hpt"],
    3: ["hpc", "hpt", "combustor"],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Model definition (must match training architecture exactly)
# ---------------------------------------------------------------------------

class FaultClassifier1DCNN(nn.Module):
    """Wider 3-conv-layer 1D-CNN, matches the unified classifier (Phase 13)."""

    def __init__(self, input_dim, seq_len, num_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Load model + scalers at startup
# ---------------------------------------------------------------------------

model = None
global_scaler = None
engine_scalers = None
model_loaded = False
load_error = None

try:
    model = FaultClassifier1DCNN(INPUT_DIM, SEQ_LEN, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "unified_classifier_1dcnn.pt", map_location=DEVICE)
    )
    model.eval()

    # BUG-015 fix: global_scaler is required for production inference.
    # The API will refuse to scale uploaded data if it is missing.
    gs_path = MODEL_DIR / "global_scaler.pkl"
    if gs_path.exists():
        global_scaler = joblib.load(gs_path)
    else:
        import warnings
        warnings.warn(
            f"global_scaler.pkl not found at {gs_path}. "
            "Inference requests will be rejected until the scaler is present.",
            RuntimeWarning,
            stacklevel=1,
        )

    model_loaded = True
except Exception as exc:  # noqa: BLE001
    load_error = str(exc)
    model_loaded = False


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def normalize_window(df: pd.DataFrame) -> np.ndarray:
    """
    Normalize a single engine's window of raw sensor readings.
    Expects columns matching FEATURE_COLS (extra columns are ignored).

    BUG-015 fix: strictly uses the persisted global_scaler for all features.
    Raises RuntimeError if the scaler is not loaded, rather than silently
    refitting on uploaded data (which would be per-request information leakage).
    """
    if global_scaler is None:
        raise RuntimeError(
            "global_scaler is not loaded. Place global_scaler.pkl in "
            f"{MODEL_DIR} and restart the server before running inference."
        )

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[FEATURE_COLS].copy()

    try:
        data[FEATURE_COLS] = global_scaler.transform(data[FEATURE_COLS])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Scaler transform failed: {exc}") from exc

    return data[FEATURE_COLS].values.astype(np.float32)


def build_window(values: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Take the last seq_len rows, zero-padding from the front if too short."""
    if len(values) >= seq_len:
        return values[-seq_len:]
    pad = np.zeros((seq_len - len(values), values.shape[1]), dtype=np.float32)
    return np.vstack([pad, values])


def run_inference(window: np.ndarray) -> dict:
    """window shape: (seq_len, input_dim) -> prediction dict."""
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    rul_estimate = CLASS_RUL_MIDPOINT[pred_class]

    return {
        "predicted_class": pred_class,
        "class_name": CLASS_NAMES[pred_class],
        "confidence": round(float(probs[pred_class]), 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
        "rul_estimate": rul_estimate,
        "fault_zones": FAULT_ZONE_MAP[pred_class],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok" if model_loaded else "model_not_loaded",
            "device": str(DEVICE),
            "model_loaded": model_loaded,
            "error": load_error,
        }
    )


@app.route("/api/classes", methods=["GET"])
def classes():
    return jsonify(
        {
            "classes": [
                {
                    "id": i,
                    "name": CLASS_NAMES[i],
                    "rul_midpoint": CLASS_RUL_MIDPOINT[i],
                    "fault_zones": FAULT_ZONE_MAP[i],
                }
                for i in range(NUM_CLASSES)
            ],
            "feature_cols": FEATURE_COLS,
            "seq_len": SEQ_LEN,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": f"Model not loaded: {load_error}"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        raw_bytes = file.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

    try:
        normalized = normalize_window(df)
        window = build_window(normalized)
        result = run_inference(window)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    result["rows_received"] = int(len(df))
    result["window_used"] = SEQ_LEN
    return jsonify(result)


@app.route("/api/sample/<int:class_id>", methods=["GET"])
def sample(class_id: int):
    """
    Returns a synthetic 30-step window for demo purposes (no model needed),
    useful for the frontend's "try a sample" buttons without requiring a CSV.
    """
    if class_id not in CLASS_RUL_MIDPOINT:
        return jsonify({"error": "class_id must be 0-3"}), 400

    rng = np.random.default_rng(seed=class_id * 17 + 3)
    base_level = {0: 0.3, 1: 0.45, 2: 0.6, 3: 0.78}[class_id]
    trend = {0: 0.0, 1: 0.003, 2: 0.006, 3: 0.012}[class_id]

    series = []
    for t in range(SEQ_LEN):
        row = base_level + trend * t + rng.normal(0, 0.05, size=INPUT_DIM)
        series.append(np.clip(row, 0, 1))
    series = np.array(series, dtype=np.float32)

    result = {
        "predicted_class": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": 1.0,
        "probabilities": {
            CLASS_NAMES[i]: (1.0 if i == class_id else 0.0)
            for i in range(NUM_CLASSES)
        },
        "rul_estimate": CLASS_RUL_MIDPOINT[class_id],
        "fault_zones": FAULT_ZONE_MAP[class_id],
    }
    result["window"] = series.tolist()
    result["feature_cols"] = FEATURE_COLS
    return jsonify(result)


if __name__ == "__main__":
    print(f"Model loaded: {model_loaded}")
    if not model_loaded:
        print(f"Load error: {load_error}")
    app.run(debug=True, host="0.0.0.0", port=5000)
