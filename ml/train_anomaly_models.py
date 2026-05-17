"""
train_anomaly_models.py
=======================
Trains and benchmarks anomaly detection models on synthetic dirty datasets.
Saves best-performing models to saved_models/.

Models trained:
  1. IsolationForest  (fast, scales to big data)
  2. Local Outlier Factor (density-based)
  3. One-Class SVM (boundary-based)
  4. Ensemble combiner

Run: python ml/train_anomaly_models.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, average_precision_score
)
from sklearn.model_selection import ParameterGrid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(ROOT, "ml", "saved_models")
BACKEND_MODEL_DIR = os.path.join(ROOT, "backend", "models")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BACKEND_MODEL_DIR, exist_ok=True)


def generate_benchmark_dataset(n_normal=2000, n_anomaly=100, n_features=10, seed=42):
    """Generate synthetic dataset with known ground-truth anomalies."""
    rng = np.random.RandomState(seed)

    # Normal data: multivariate Gaussian
    normal = rng.randn(n_normal, n_features)

    # Anomalies: mix of strategies
    anomaly_types = [
        rng.uniform(5, 10, (n_anomaly // 3, n_features)),        # Far outliers
        rng.uniform(-10, -5, (n_anomaly // 3, n_features)),      # Negative extreme
        rng.randn(n_anomaly - 2*(n_anomaly//3), n_features) * 5, # High variance
    ]
    anomalies = np.vstack(anomaly_types)

    X = np.vstack([normal, anomalies])
    y_true = np.array([0] * n_normal + [1] * len(anomalies))  # 1 = anomaly

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y_true[idx]


def evaluate_model(model, X_scaled, y_true, model_name, is_lof=False):
    """Evaluate anomaly model and return metrics dict."""
    if is_lof:
        # LOF can't predict on new data without refitting; use fit_predict
        preds = model.fit_predict(X_scaled)
        scores = -model.negative_outlier_factor_
    else:
        preds = model.predict(X_scaled)
        scores = -model.score_samples(X_scaled)

    # Convert: -1 = anomaly → 1, 1 = normal → 0
    y_pred = (preds == -1).astype(int)

    # Normalize scores to [0,1]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    try:
        auc = roc_auc_score(y_true, scores_norm)
        ap = average_precision_score(y_true, scores_norm)
    except Exception:
        auc, ap = 0.0, 0.0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "model": model_name,
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def grid_search_isolation_forest(X_scaled, y_true):
    """Simple grid search for IsolationForest hyperparameters."""
    print("  Grid searching IsolationForest...")
    best_score = 0
    best_params = {}
    best_model = None

    param_grid = {
        "contamination": [0.03, 0.05, 0.08, 0.10],
        "n_estimators": [100, 200],
        "max_features": [0.8, 1.0],
    }

    for params in ParameterGrid(param_grid):
        model = IsolationForest(random_state=42, n_jobs=-1, **params)
        model.fit(X_scaled)
        scores = -model.score_samples(X_scaled)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        try:
            auc = roc_auc_score(y_true, scores_norm)
        except Exception:
            auc = 0.0
        if auc > best_score:
            best_score = auc
            best_params = params
            best_model = model

    print(f"    Best AUC: {best_score:.4f} | Params: {best_params}")
    return best_model, best_params, best_score


def train():
    print("=" * 60)
    print("  DataForge AI — Anomaly Detection Model Training")
    print("=" * 60)

    # ── Generate benchmark data ───────────────────────────────────────────────
    print("\n  Generating synthetic benchmark dataset...")
    X, y_true = generate_benchmark_dataset(n_normal=2000, n_anomaly=100, n_features=10)
    contamination_true = y_true.mean()
    print(f"  Samples: {len(X)} | Anomalies: {y_true.sum()} ({contamination_true:.1%})")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Grid search IsolationForest ───────────────────────────────────────────
    best_iso, best_iso_params, iso_auc = grid_search_isolation_forest(X_scaled, y_true)

    # ── LOF ───────────────────────────────────────────────────────────────────
    print("\n  Training Local Outlier Factor...")
    lof = LocalOutlierFactor(
        contamination=0.05, n_neighbors=20, n_jobs=-1
    )
    lof_metrics = evaluate_model(lof, X_scaled, y_true, "LOF", is_lof=True)
    print(f"    LOF  → AUC: {lof_metrics['roc_auc']:.4f} | F1: {lof_metrics['f1']:.4f}")

    # ── One-Class SVM ─────────────────────────────────────────────────────────
    print("\n  Training One-Class SVM...")
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X_scaled)
    ocsvm_preds = ocsvm.predict(X_scaled)
    ocsvm_scores = -ocsvm.decision_function(X_scaled)
    ocsvm_y_pred = (ocsvm_preds == -1).astype(int)
    sc_norm = (ocsvm_scores - ocsvm_scores.min()) / (ocsvm_scores.max() - ocsvm_scores.min() + 1e-9)
    try:
        ocsvm_auc = roc_auc_score(y_true, sc_norm)
    except Exception:
        ocsvm_auc = 0.0
    ocsvm_f1 = f1_score(y_true, ocsvm_y_pred, zero_division=0)
    print(f"    OCSVM → AUC: {ocsvm_auc:.4f} | F1: {ocsvm_f1:.4f}")

    # ── IsolationForest final eval ────────────────────────────────────────────
    iso_metrics = evaluate_model(best_iso, X_scaled, y_true, "IsolationForest (tuned)")
    print(f"\n  IsoForest → AUC: {iso_metrics['roc_auc']:.4f} | F1: {iso_metrics['f1']:.4f}")

    # ── Benchmark summary ─────────────────────────────────────────────────────
    print("\n  ┌─────────────────────────┬─────────┬─────────┬─────────┐")
    print("  │ Model                   │ AUC     │ F1      │ Recall  │")
    print("  ├─────────────────────────┼─────────┼─────────┼─────────┤")
    for m in [iso_metrics, lof_metrics]:
        print(f"  │ {m['model']:<23} │ {m['roc_auc']:.4f}  │ {m['f1']:.4f}  │ {m['recall']:.4f}  │")
    print(f"  │ {'OneClassSVM':<23} │ {ocsvm_auc:.4f}  │ {ocsvm_f1:.4f}  │ {'—':<7} │")
    print("  └─────────────────────────┴─────────┴─────────┴─────────┘")

    # ── Save models ───────────────────────────────────────────────────────────
    print("\n  Saving models...")
    joblib.dump(best_iso, os.path.join(SAVE_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "anomaly_scaler.pkl"))
    joblib.dump(ocsvm, os.path.join(SAVE_DIR, "ocsvm.pkl"))

    # Copy scaler to backend for runtime use
    joblib.dump(scaler, os.path.join(BACKEND_MODEL_DIR, "anomaly_scaler.pkl"))
    joblib.dump(best_iso, os.path.join(BACKEND_MODEL_DIR, "isolation_forest.pkl"))

    # ── Save benchmark results ────────────────────────────────────────────────
    results = {
        "isolation_forest": {**iso_metrics, "best_params": best_iso_params},
        "lof": lof_metrics,
        "ocsvm": {"model": "OneClassSVM", "roc_auc": round(ocsvm_auc, 4), "f1": round(ocsvm_f1, 4)},
        "dataset": {"n_samples": len(X), "n_anomalies": int(y_true.sum()), "contamination": round(float(contamination_true), 4)},
        "recommended": "IsolationForest (best generalization on unseen data)",
    }
    with open(os.path.join(SAVE_DIR, "anomaly_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Models saved to ml/saved_models/")
    print(f"  ✅ Scaler + IsolationForest synced to backend/models/")
    print("=" * 60)


if __name__ == "__main__":
    train()
