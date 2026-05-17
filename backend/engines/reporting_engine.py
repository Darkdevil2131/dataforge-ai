"""
Reporting Engine
Generates structured quality reports with before/after stats,
transformation summaries, and risk assessments.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any


class ReportingEngine:
    def generate(
        self,
        df_original: pd.DataFrame,
        df_cleaned: pd.DataFrame,
        df_engineered: pd.DataFrame,
        schema: dict,
        intent: str,
        policy: dict,
        transformation_log: list,
        feature_log: list,
        anomaly_result: dict,
        quality_scores: dict,
    ) -> dict:
        """Generate full structured pipeline report."""

        now = datetime.utcnow().isoformat() + "Z"

        # ── Column-level summary ──────────────────────────────────────────
        column_summary = []
        for col in df_original.columns:
            clean_col = col.strip().lower().replace(" ", "_").replace("-", "_")
            clean_col = "".join(c if c.isalnum() or c == "_" else "_" for c in clean_col)
            info = schema.get(clean_col, schema.get(col, {}))
            col_summary = {
                "original_name": col,
                "clean_name": clean_col,
                "dtype": info.get("dtype", "unknown"),
                "role": info.get("role", "unknown"),
                "inferred_type": info.get("inferred_type", "unknown"),
                "null_count_before": int(df_original[col].isnull().sum()),
                "null_pct_before": round(df_original[col].isnull().mean() * 100, 2),
                "unique_before": int(df_original[col].nunique()),
            }
            if clean_col in df_cleaned.columns:
                col_summary["null_count_after"] = int(df_cleaned[clean_col].isnull().sum())
                col_summary["null_pct_after"] = round(df_cleaned[clean_col].isnull().mean() * 100, 2)
                col_summary["unique_after"] = int(df_cleaned[clean_col].nunique())
            column_summary.append(col_summary)

        # ── Transformation summary ────────────────────────────────────────
        auto_applied = [t for t in transformation_log if t["status"] == "auto_applied"]
        suggested = [t for t in transformation_log if t["status"] == "suggested"]
        flagged = [t for t in transformation_log if t["status"] == "flagged"]

        # ── Risk assessment ───────────────────────────────────────────────
        risks = []
        if policy.get("preserve_outliers") is False and anomaly_result.get("anomaly_count", 0) > 0:
            risks.append({
                "severity": "medium",
                "message": f"Outlier removal affected {anomaly_result['anomaly_count']} rows.",
                "recommendation": "Verify that removed outliers were not domain-significant."
            })

        if quality_scores.get("before", 0) < 60:
            risks.append({
                "severity": "high",
                "message": f"Dataset quality was low ({quality_scores['before']}%) before cleaning.",
                "recommendation": "Consider collecting more or better quality data."
            })

        if df_original.shape[0] < 100:
            risks.append({
                "severity": "high",
                "message": f"Dataset is very small ({df_original.shape[0]} rows).",
                "recommendation": "ML models may underfit. Consider data augmentation or a larger dataset."
            })

        if df_engineered.shape[1] > df_original.shape[1] * 3:
            risks.append({
                "severity": "medium",
                "message": f"Feature explosion: {df_original.shape[1]} → {df_engineered.shape[1]} features.",
                "recommendation": "Apply PCA or feature selection before modeling."
            })

        # ── Model recommendations ─────────────────────────────────────────
        model_recs = self._get_model_recommendations(
            intent, df_engineered, anomaly_result
        )

        return {
            "generated_at": now,
            "intent": intent,
            "cleaning_policy": policy.get("description", ""),
            "dataset_shape": {
                "original": list(df_original.shape),
                "cleaned": list(df_cleaned.shape),
                "engineered": list(df_engineered.shape),
            },
            "quality_scores": quality_scores,
            "column_summary": column_summary,
            "transformations": {
                "auto_applied": auto_applied,
                "suggested": suggested,
                "flagged": flagged,
                "total": len(transformation_log)
            },
            "feature_engineering": {
                "features_added": len(feature_log),
                "details": feature_log
            },
            "anomaly_detection": {
                "anomalies_found": anomaly_result.get("anomaly_count", 0),
                "suspected": anomaly_result.get("suspected_count", 0),
                "preserved": anomaly_result.get("preserved", False),
                "method": anomaly_result.get("method", "none"),
                "details": anomaly_result.get("anomaly_details", [])[:10]
            },
            "risks": risks,
            "model_recommendations": model_recs,
            "policy_notes": policy.get("risk_notes", []),
            "next_steps": self._get_next_steps(intent, quality_scores, df_engineered)
        }

    def _get_model_recommendations(self, intent: str, df: pd.DataFrame,
                                    anomaly_result: dict) -> list:
        """Recommend models based on intent and data characteristics."""
        recs = {
            "fraud_detection": [
                {"model": "XGBoost", "why": "Best AUC on imbalanced tabular data", "priority": 1},
                {"model": "LightGBM", "why": "Fast, handles class imbalance", "priority": 2},
                {"model": "IsolationForest", "why": "Unsupervised — no labels needed", "priority": 3},
            ],
            "classification": [
                {"model": "Random Forest", "why": "Robust baseline, feature importance", "priority": 1},
                {"model": "XGBoost", "why": "State of the art for tabular data", "priority": 2},
                {"model": "Logistic Regression", "why": "Interpretable, good for baseline", "priority": 3},
            ],
            "regression": [
                {"model": "XGBoost Regressor", "why": "Best tabular regression performance", "priority": 1},
                {"model": "Ridge / Lasso", "why": "Fast, regularized, interpretable", "priority": 2},
                {"model": "Random Forest Regressor", "why": "Handles non-linearity", "priority": 3},
            ],
            "forecasting": [
                {"model": "Prophet", "why": "Handles seasonality + trends automatically", "priority": 1},
                {"model": "XGBoost + lag features", "why": "Often outperforms LSTM on tabular", "priority": 2},
                {"model": "ARIMA", "why": "Classical, interpretable baseline", "priority": 3},
            ],
            "analytics": [
                {"model": "No ML needed", "why": "Use aggregation + visualization", "priority": 1},
                {"model": "K-Means Clustering", "why": "Customer segmentation", "priority": 2},
                {"model": "PCA", "why": "Dimensionality reduction for exploration", "priority": 3},
            ],
            "recommendation": [
                {"model": "SVD (Surprise library)", "why": "Collaborative filtering standard", "priority": 1},
                {"model": "ALS", "why": "Scalable matrix factorization", "priority": 2},
                {"model": "LightFM", "why": "Hybrid CF + content", "priority": 3},
            ],
            "nlp": [
                {"model": "TF-IDF + Logistic Regression", "why": "Strong baseline, fast", "priority": 1},
                {"model": "BERT (distilbert)", "why": "State of the art NLP", "priority": 2},
                {"model": "FastText", "why": "Lightweight, production-ready", "priority": 3},
            ],
            "general_ml": [
                {"model": "Random Forest", "why": "Good all-round baseline", "priority": 1},
                {"model": "XGBoost", "why": "Industry standard", "priority": 2},
                {"model": "Logistic Regression", "why": "Simple, interpretable", "priority": 3},
            ]
        }
        return recs.get(intent, recs["general_ml"])

    def _get_next_steps(self, intent: str, quality: dict, df: pd.DataFrame) -> list:
        """Generate actionable next steps."""
        steps = [f"✅ Dataset cleaned — quality score improved from {quality.get('before', 0)}% to {quality.get('after', 0)}%"]

        if intent in ("classification", "regression", "fraud_detection"):
            steps.append("📊 Split data: 70% train / 15% validation / 15% test")
            steps.append("🔍 Run feature importance to eliminate low-signal features")

        if intent == "forecasting":
            steps.append("📅 Sort by timestamp before train/test split — never shuffle")
            steps.append("📈 Check for stationarity (ADF test) before ARIMA")

        if intent == "analytics":
            steps.append("📊 Load into Power BI / Tableau / Apache Superset")
            steps.append("🗂️ Create date dimension table for drill-down analysis")

        steps.append("💾 Download cleaned CSV and use it directly for training")
        steps.append("📄 Review transformation log for any suggested (unconfirmed) changes")

        return steps
