"""
Automated Feature Engineering Engine
Generates features based on intent:
- Time-based features for forecasting
- Interaction features for classification/regression
- Encoding for categoricals
- Normalization for numeric
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Any
import warnings
warnings.filterwarnings("ignore")


class FeatureEngine:
    def __init__(self):
        self.fitted_scalers = {}
        self.fitted_encoders = {}
        self.feature_log = []

    def engineer(self, df: pd.DataFrame, schema: dict,
                 intent: str, policy: dict) -> tuple[pd.DataFrame, list]:
        """
        Apply feature engineering based on detected intent.
        Returns (engineered_df, feature_log)
        """
        df = df.copy()
        self.feature_log = []

        # Identify column types from schema
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []

        for col in df.columns:
            info = schema.get(col, {})
            role = info.get("role", "unknown")
            inferred = info.get("inferred_type", "unknown")

            if role == "identifier":
                continue  # Skip IDs

            if inferred in ("float", "integer", "binary_flag") or pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() > 2:
                    numeric_cols.append(col)
            elif inferred == "categorical" or (df[col].dtype == object and df[col].nunique() <= 50):
                categorical_cols.append(col)
            elif inferred in ("datetime", "datetime_string") or pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)

        # ── 1. DateTime Features ─────────────────────────────────────────
        for col in datetime_cols:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                dt = df[col]
                df[f"{col}_year"] = dt.dt.year
                df[f"{col}_month"] = dt.dt.month
                df[f"{col}_day"] = dt.dt.day
                df[f"{col}_dayofweek"] = dt.dt.dayofweek
                df[f"{col}_quarter"] = dt.dt.quarter
                df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

                if intent in ("forecasting", "analytics"):
                    df[f"{col}_dayofyear"] = dt.dt.dayofyear
                    # Cyclical encoding for month and hour
                    df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
                    df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

                self.feature_log.append({
                    "action": "extract_datetime_features",
                    "source_column": col,
                    "features_created": [f"{col}_year", f"{col}_month", f"{col}_day",
                                         f"{col}_dayofweek", f"{col}_quarter", f"{col}_is_weekend"],
                    "reason": f"DateTime column '{col}' decomposed into temporal features"
                })
            except Exception as e:
                pass

        # ── 2. Lag + Rolling Features (forecasting only) ─────────────────
        if intent == "forecasting":
            for col in numeric_cols[:3]:  # Limit to first 3 numeric to avoid explosion
                try:
                    df[f"{col}_lag_1"] = df[col].shift(1)
                    df[f"{col}_lag_7"] = df[col].shift(7)
                    df[f"{col}_rolling_mean_7"] = df[col].rolling(window=7, min_periods=1).mean()
                    df[f"{col}_rolling_std_7"] = df[col].rolling(window=7, min_periods=1).std()
                    self.feature_log.append({
                        "action": "create_lag_rolling_features",
                        "source_column": col,
                        "features_created": [f"{col}_lag_1", f"{col}_lag_7",
                                             f"{col}_rolling_mean_7", f"{col}_rolling_std_7"],
                        "reason": "Forecasting task — lag and rolling window features added"
                    })
                except Exception:
                    pass

        # ── 3. Categorical Encoding ──────────────────────────────────────
        for col in categorical_cols:
            if col not in df.columns:
                continue
            n_unique = df[col].nunique()

            if n_unique == 2:
                # Binary encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.fitted_encoders[col] = ("label", le)
                self.feature_log.append({
                    "action": "binary_label_encode",
                    "source_column": col,
                    "features_created": [col],
                    "reason": f"Binary categorical ({n_unique} unique values)"
                })
            elif n_unique <= 15:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
                self.feature_log.append({
                    "action": "one_hot_encode",
                    "source_column": col,
                    "features_created": list(dummies.columns),
                    "reason": f"Categorical with {n_unique} unique values → one-hot encoded"
                })
            else:
                # Frequency encoding for high-cardinality
                freq = df[col].value_counts(normalize=True)
                df[f"{col}_freq_enc"] = df[col].map(freq).fillna(0)
                df = df.drop(columns=[col])
                self.feature_log.append({
                    "action": "frequency_encode",
                    "source_column": col,
                    "features_created": [f"{col}_freq_enc"],
                    "reason": f"High-cardinality categorical ({n_unique} unique) → frequency encoded"
                })

        # ── 4. Numeric Scaling ───────────────────────────────────────────
        # Refresh numeric cols after encoding changes
        numeric_cols_final = [c for c in df.columns
                               if pd.api.types.is_numeric_dtype(df[c])
                               and df[c].nunique() > 2
                               and not c.endswith("_freq_enc")]

        if intent in ("classification", "regression", "fraud_detection", "nlp"):
            if policy.get("normalize_skew", True):
                scaled_cols = []
                scaler = StandardScaler()
                cols_to_scale = [c for c in numeric_cols_final
                                  if c in df.columns and not c.startswith("is_")]
                if cols_to_scale:
                    try:
                        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].fillna(0))
                        self.fitted_scalers["standard"] = scaler
                        scaled_cols = cols_to_scale
                        self.feature_log.append({
                            "action": "standard_scale",
                            "source_column": "MULTIPLE",
                            "features_created": scaled_cols,
                            "reason": f"StandardScaler applied to {len(scaled_cols)} numeric features for {intent}"
                        })
                    except Exception:
                        pass

        return df, self.feature_log

    def detect_feature_importance(self, df: pd.DataFrame,
                                   target_col: str = None) -> list:
        """Quick variance-based feature importance (no model needed)."""
        if target_col and target_col in df.columns:
            # Correlation with target
            numeric_df = df.select_dtypes(include=[np.number])
            if target_col in numeric_df.columns:
                corr = numeric_df.corr()[target_col].drop(target_col, errors="ignore")
                corr = corr.abs().sort_values(ascending=False)
                return [{"feature": k, "importance_score": round(float(v), 4)}
                        for k, v in corr.head(20).items()]
        # Variance-based
        numeric_df = df.select_dtypes(include=[np.number])
        var = numeric_df.var().sort_values(ascending=False)
        return [{"feature": k, "importance_score": round(float(v), 4)}
                for k, v in var.head(20).items()]
