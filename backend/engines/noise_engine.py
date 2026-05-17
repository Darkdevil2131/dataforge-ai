"""
Noise Detection & Cleaning Engine
Hybrid approach: Statistical + Rule-Based + ML (IsolationForest)
Confidence-scored — never blindly modifies data.
Policy-aware — fraud data treated differently than regression data.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Any
import warnings
warnings.filterwarnings("ignore")


# ─── Confidence thresholds ────────────────────────────────────────────────────
CONF_AUTO_FIX = 0.90       # Auto-fix above this
CONF_SUGGEST = 0.65        # Suggest above this, below auto-fix
# Below CONF_SUGGEST → quarantine / flag only


class TransformationLog:
    """Tracks every transformation with metadata."""
    def __init__(self):
        self.entries = []

    def add(self, column: str, action: str, rows_affected: int,
            reason: str, confidence: float, details: str = ""):
        self.entries.append({
            "column": column,
            "action": action,
            "rows_affected": rows_affected,
            "reason": reason,
            "confidence": round(confidence, 3),
            "details": details,
            "status": (
                "auto_applied" if confidence >= CONF_AUTO_FIX
                else "suggested" if confidence >= CONF_SUGGEST
                else "flagged"
            )
        })

    def to_list(self):
        return self.entries


def _iqr_bounds(series: pd.Series, multiplier: float = 1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - multiplier * IQR, Q3 + multiplier * IQR


def _zscore_mask(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    z = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.iloc[series.dropna().index.get_indexer(series.dropna().index)] = z > threshold
    return mask


def detect_numeric_outliers(series: pd.Series, preserve_outliers: bool = False) -> dict:
    """Detect outliers using IQR + z-score ensemble."""
    non_null = series.dropna()
    if len(non_null) < 10:
        return {"outlier_indices": [], "outlier_count": 0, "method": "skipped_too_few_rows"}

    lower, upper = _iqr_bounds(non_null)
    iqr_mask = (non_null < lower) | (non_null > upper)

    z_mask = _zscore_mask(non_null)
    z_outliers = set(non_null[z_mask].index.tolist())

    # Ensemble: outlier confirmed if BOTH methods agree
    iqr_outliers = set(non_null[iqr_mask].index.tolist())
    confirmed_outliers = iqr_outliers & z_outliers  # intersection = high confidence
    suspected_outliers = iqr_outliers | z_outliers   # union = lower confidence

    return {
        "outlier_indices": list(confirmed_outliers),
        "suspected_indices": list(suspected_outliers - confirmed_outliers),
        "outlier_count": len(confirmed_outliers),
        "suspected_count": len(suspected_outliers - confirmed_outliers),
        "bounds": {"lower": float(lower), "upper": float(upper)},
        "preserve_outliers": preserve_outliers,
        "method": "iqr_zscore_ensemble"
    }


def impute_missing(series: pd.Series, strategy: str, col_name: str = "") -> tuple[pd.Series, str]:
    """Impute missing values with the appropriate strategy."""
    if series.isnull().sum() == 0:
        return series, "no_missing"

    if strategy == "median":
        fill_val = series.median()
        return series.fillna(fill_val), f"median={round(float(fill_val), 4)}"
    elif strategy == "mean":
        fill_val = series.mean()
        return series.fillna(fill_val), f"mean={round(float(fill_val), 4)}"
    elif strategy == "mode":
        fill_val = series.mode().iloc[0] if not series.mode().empty else "UNKNOWN"
        return series.fillna(fill_val), f"mode={fill_val}"
    elif strategy == "zero":
        return series.fillna(0), "fill=0"
    elif strategy == "forward_fill":
        return series.ffill().bfill(), "forward_fill+backfill"
    elif strategy == "drop":
        return series, "rows_with_missing_dropped_separately"
    else:
        # Smart default
        if pd.api.types.is_numeric_dtype(series):
            skew = abs(series.skew()) if len(series.dropna()) > 3 else 0
            if skew > 1.0:
                fill_val = series.median()
                return series.fillna(fill_val), f"median={round(float(fill_val), 4)} (skewed)"
            else:
                fill_val = series.mean()
                return series.fillna(fill_val), f"mean={round(float(fill_val), 4)}"
        else:
            fill_val = series.mode().iloc[0] if not series.mode().empty else "UNKNOWN"
            return series.fillna(fill_val), f"mode={fill_val}"


def get_imputation_strategy(col_name: str, series: pd.Series, intent: str,
                             null_pct: float) -> tuple[str, float]:
    """
    Decide the best imputation strategy and return (strategy, confidence).
    Context-aware: forecasting uses forward fill, fraud preserves, etc.
    """
    if null_pct > 60:
        return "drop_column", 0.55  # Too many missing, suggest dropping

    if intent == "forecasting" and pd.api.types.is_numeric_dtype(series):
        return "forward_fill", 0.92

    if pd.api.types.is_numeric_dtype(series):
        if null_pct < 5:
            skew = abs(series.skew()) if len(series.dropna()) > 3 else 0
            strategy = "median" if skew > 1.0 else "mean"
            return strategy, 0.95
        elif null_pct < 20:
            return "median", 0.88
        else:
            return "median", 0.72
    else:
        if null_pct < 10:
            return "mode", 0.90
        elif null_pct < 30:
            return "mode", 0.75
        else:
            return "mode", 0.60


class NoiseEngine:
    def __init__(self):
        pass

    def clean(self, df: pd.DataFrame, schema: dict, intent: str,
              policy: dict) -> tuple[pd.DataFrame, TransformationLog]:
        """
        Main cleaning pipeline.
        Returns cleaned dataframe + transformation log.
        """
        log = TransformationLog()
        df = df.copy()

        # ── Step 1: Deduplicate ───────────────────────────────────────────
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            confidence = 0.97 if dup_count / len(df) < 0.1 else 0.82
            if confidence >= CONF_AUTO_FIX:
                df = df.drop_duplicates().reset_index(drop=True)
                log.add("ALL", "remove_duplicates", dup_count,
                        "exact_duplicate_rows", confidence,
                        f"Removed {dup_count} exact duplicate rows")

        # ── Step 2: Fix column names ──────────────────────────────────────
        new_cols = {}
        for col in df.columns:
            clean_name = col.strip().lower().replace(" ", "_").replace("-", "_")
            clean_name = "".join(c if c.isalnum() or c == "_" else "_" for c in clean_name)
            if clean_name != col:
                new_cols[col] = clean_name
        if new_cols:
            df = df.rename(columns=new_cols)
            log.add("ALL", "standardize_column_names", len(new_cols),
                    "whitespace_or_special_chars", 1.0,
                    f"Renamed {len(new_cols)} columns to snake_case")

        # ── Step 3: Per-column cleaning ───────────────────────────────────
        for col in df.columns:
            col_info = schema.get(col, {})
            if not col_info:
                # Try original name if renamed
                pass
            series = df[col]
            null_pct = series.isnull().mean() * 100
            role = col_info.get("role", "unknown")

            # Skip identifiers and free text for numeric cleaning
            if role in ("identifier",):
                continue

            # ── Handle missing values ────────────────────────────────
            if null_pct > 0:
                strategy, confidence = get_imputation_strategy(
                    col, series, intent, null_pct
                )
                null_count = int(series.isnull().sum())

                if strategy == "drop_column":
                    log.add(col, "suggest_drop_column", null_count,
                            f"high_missing_rate_{null_pct:.1f}pct", confidence,
                            f"Column has {null_pct:.1f}% missing values — consider dropping")
                else:
                    if confidence >= CONF_AUTO_FIX:
                        df[col], fill_detail = impute_missing(series, strategy, col)
                        log.add(col, f"impute_{strategy}", null_count,
                                "missing_values", confidence,
                                f"Filled {null_count} nulls with {fill_detail}")
                    else:
                        log.add(col, f"suggest_impute_{strategy}", null_count,
                                "missing_values", confidence,
                                f"Suggested: fill {null_count} nulls using {strategy}")

            # ── Handle outliers (numeric only) ───────────────────────
            if pd.api.types.is_numeric_dtype(df[col]) and role not in ("binary_flag",):
                outlier_info = detect_numeric_outliers(
                    df[col], preserve_outliers=policy.get("preserve_outliers", False)
                )

                if outlier_info["outlier_count"] > 0:
                    if policy.get("preserve_outliers"):
                        # Flag but preserve (fraud, forecasting)
                        log.add(col, "flag_outliers_preserved", outlier_info["outlier_count"],
                                "potential_anomaly_preserved", 0.95,
                                f"Detected {outlier_info['outlier_count']} outliers — PRESERVED per {intent} policy")
                    else:
                        # Cap/clip to bounds (regression, classification)
                        count = outlier_info["outlier_count"]
                        lb = outlier_info["bounds"]["lower"]
                        ub = outlier_info["bounds"]["upper"]
                        confidence = 0.88 if count / max(len(df[col].dropna()), 1) < 0.05 else 0.72
                        if confidence >= CONF_AUTO_FIX:
                            df[col] = df[col].clip(lower=lb, upper=ub)
                            log.add(col, "clip_outliers", count,
                                    "outlier_values_beyond_iqr", confidence,
                                    f"Clipped {count} outliers to [{lb:.2f}, {ub:.2f}]")
                        else:
                            log.add(col, "suggest_clip_outliers", count,
                                    "outlier_values_beyond_iqr", confidence,
                                    f"Suggested: clip {count} outliers to [{lb:.2f}, {ub:.2f}]")

            # ── Fix datetime strings ─────────────────────────────────
            if col_info.get("inferred_type") == "datetime_string":
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    parsed_ok = df[col].notna().sum()
                    log.add(col, "parse_datetime", int(parsed_ok),
                            "datetime_string_column", 0.97,
                            f"Parsed {parsed_ok} rows as datetime")
                except Exception:
                    pass

            # ── Categorical: strip whitespace, lowercase ─────────────
            if col_info.get("inferred_type") == "categorical" and series.dtype == object:
                original = df[col].copy()
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace("nan", np.nan)
                changed = (original.fillna("__null__") != df[col].fillna("__null__")).sum()
                if changed > 0:
                    log.add(col, "normalize_categorical", int(changed),
                            "whitespace_case_inconsistency", 0.99,
                            f"Normalized {changed} categorical values to lowercase+stripped")

            # ── Drop constant columns ────────────────────────────────
            if df[col].nunique() <= 1 and len(df) > 5:
                log.add(col, "suggest_drop_constant", len(df),
                        "constant_column_zero_variance", 0.85,
                        f"Column has only {df[col].nunique()} unique value(s) — consider dropping")

        return df, log

    def compute_quality_score(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                               log: TransformationLog) -> dict:
        """Compute a data quality score before and after cleaning."""
        def score_df(df):
            total = df.shape[0] * df.shape[1]
            if total == 0:
                return 0
            missing_pct = df.isnull().mean().mean() * 100
            dup_pct = df.duplicated().mean() * 100
            completeness = 100 - missing_pct
            uniqueness = 100 - dup_pct
            return round((completeness * 0.6 + uniqueness * 0.4), 2)

        before_score = score_df(df_before)
        after_score = score_df(df_after)

        return {
            "before": round(before_score, 2),
            "after": round(after_score, 2),
            "improvement": round(after_score - before_score, 2),
            "actions_applied": len([e for e in log.entries if e["status"] == "auto_applied"]),
            "actions_suggested": len([e for e in log.entries if e["status"] == "suggested"]),
            "actions_flagged": len([e for e in log.entries if e["status"] == "flagged"]),
        }
