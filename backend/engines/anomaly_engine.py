"""
Anomaly Detection Engine
Uses Isolation Forest + LOF + Statistical ensemble.
No GPU needed. Runs on CPU / free tier.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class AnomalyEngine:
    def __init__(self):
        self.models = {}

    def detect(self, df: pd.DataFrame, preserve_outliers: bool = False,
               contamination: float = 0.05) -> dict:
        """
        Run ensemble anomaly detection on numeric columns.
        Returns anomaly scores and flagged indices.
        """
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        if numeric_df.shape[1] == 0:
            return {"anomaly_count": 0, "anomaly_indices": [], "method": "no_numeric_cols"}
        if len(numeric_df) < 20:
            return {"anomaly_count": 0, "anomaly_indices": [], "method": "too_few_rows"}

        # Fill remaining NaN for model
        numeric_df = numeric_df.fillna(numeric_df.median())

        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df)

        # ── IsolationForest ───────────────────────────────────────────────
        try:
            iso = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            )
            iso_preds = iso.fit_predict(X)   # -1 = anomaly
            iso_scores = iso.score_samples(X)  # more negative = more anomalous
        except Exception:
            iso_preds = np.ones(len(X))
            iso_scores = np.zeros(len(X))

        # ── Local Outlier Factor ──────────────────────────────────────────
        try:
            lof = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=min(20, len(X) - 1),
                n_jobs=-1
            )
            lof_preds = lof.fit_predict(X)  # -1 = anomaly
            lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous
        except Exception:
            lof_preds = np.ones(len(X))
            lof_scores = np.zeros(len(X))

        # ── Ensemble: flag if both agree ─────────────────────────────────
        iso_anomaly = iso_preds == -1
        lof_anomaly = lof_preds == -1

        both_agree = iso_anomaly & lof_anomaly       # High confidence
        at_least_one = iso_anomaly | lof_anomaly     # Medium confidence

        # Normalize scores to [0, 1]
        def normalize(arr):
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        iso_norm = normalize(-iso_scores)  # flip: higher = more anomalous
        lof_norm = normalize(lof_scores)
        ensemble_score = (iso_norm + lof_norm) / 2

        confirmed_indices = list(np.where(both_agree)[0])
        suspected_indices = list(np.where(at_least_one & ~both_agree)[0])

        # Build per-row anomaly details
        anomaly_details = []
        for idx in confirmed_indices[:50]:  # Cap at 50 for response size
            row_data = {}
            for col in numeric_df.columns:
                row_data[col] = round(float(numeric_df.iloc[idx][col]), 4)
            anomaly_details.append({
                "row_index": int(idx),
                "anomaly_score": round(float(ensemble_score[idx]), 4),
                "confidence": "high",
                "data": row_data
            })

        return {
            "anomaly_count": len(confirmed_indices),
            "suspected_count": len(suspected_indices),
            "total_rows": len(df),
            "anomaly_pct": round(len(confirmed_indices) / max(len(df), 1) * 100, 2),
            "anomaly_indices": confirmed_indices,
            "suspected_indices": suspected_indices[:20],
            "anomaly_details": anomaly_details,
            "method": "isolation_forest_lof_ensemble",
            "preserved": preserve_outliers,
            "note": (
                "Anomalies preserved as per fraud/forecasting policy — these are valuable signals."
                if preserve_outliers
                else "Anomalies detected — consider reviewing before removal."
            )
        }
