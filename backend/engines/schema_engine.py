"""
Schema Intelligence Engine
Infers column types, semantic roles, relationships, and dataset classification.
Pure rule-based + statistical. No API needed.
"""
import re
import pandas as pd
import numpy as np
from typing import Any


# ── Keyword maps for semantic column type detection ─────────────────────────
ID_PATTERNS = re.compile(
    r'\b(id|uuid|guid|key|code|ref|no|num|number|index)\b', re.IGNORECASE
)
TARGET_PATTERNS = re.compile(
    r'\b(target|label|class|output|result|outcome|y|churn|fraud|survived|survived|price|'
    r'salary|revenue|default|spam|diagnosis|disease|death|status)\b', re.IGNORECASE
)
TIMESTAMP_PATTERNS = re.compile(
    r'\b(date|time|datetime|timestamp|created|updated|at|day|month|year|period|'
    r'week|quarter|hour|minute|second)\b', re.IGNORECASE
)
TEXT_PATTERNS = re.compile(
    r'\b(name|description|comment|review|text|message|note|title|subject|content|'
    r'address|feedback|reason|remark|narrative)\b', re.IGNORECASE
)
EMAIL_PATTERNS = re.compile(r'\b(email|mail|e_mail)\b', re.IGNORECASE)
PHONE_PATTERNS = re.compile(r'\b(phone|mobile|tel|fax|contact)\b', re.IGNORECASE)
GEO_PATTERNS = re.compile(
    r'\b(lat|lon|longitude|latitude|city|country|state|zip|postal|region|location|'
    r'address|geo)\b', re.IGNORECASE
)
AMOUNT_PATTERNS = re.compile(
    r'\b(amount|price|cost|fee|salary|wage|revenue|sales|income|balance|'
    r'payment|value|total|subtotal|tax|discount)\b', re.IGNORECASE
)
LEAKAGE_PATTERNS = re.compile(
    r'\b(future|post|after|next|result|outcome|label|target|prediction|predicted|score)\b',
    re.IGNORECASE
)


def infer_column_role(col_name: str, series: pd.Series) -> str:
    """Infer the semantic role of a column."""
    name = col_name.lower()

    if ID_PATTERNS.search(name) and series.nunique() / max(len(series), 1) > 0.8:
        return "identifier"
    if TARGET_PATTERNS.search(name):
        return "potential_target"
    if TIMESTAMP_PATTERNS.search(name):
        return "temporal"
    if EMAIL_PATTERNS.search(name):
        return "email"
    if PHONE_PATTERNS.search(name):
        return "phone"
    if GEO_PATTERNS.search(name):
        return "geographic"
    if AMOUNT_PATTERNS.search(name):
        return "monetary"
    if TEXT_PATTERNS.search(name) and series.dtype == object:
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len > 30:
            return "free_text"
        return "categorical"

    # Fallback by dtype
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"
    if series.dtype == bool:
        return "boolean"
    if series.dtype == object:
        n_unique = series.nunique()
        if n_unique <= 20:
            return "categorical"
        return "text"
    return "unknown"


def detect_column_type(series: pd.Series) -> dict:
    """Detect data type and statistical properties of a column."""
    result = {
        "dtype": str(series.dtype),
        "inferred_type": "unknown",
        "nullable": series.isnull().any(),
        "null_count": int(series.isnull().sum()),
        "null_pct": round(series.isnull().mean() * 100, 2),
        "unique_count": int(series.nunique()),
        "unique_pct": round(series.nunique() / max(len(series), 1) * 100, 2),
    }

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            result["inferred_type"] = "numeric_empty"
        else:
            result["inferred_type"] = "float" if series.dtype in [np.float32, np.float64] else "integer"
            result["stats"] = {
                "mean": round(float(non_null.mean()), 4),
                "median": round(float(non_null.median()), 4),
                "std": round(float(non_null.std()), 4),
                "min": round(float(non_null.min()), 4),
                "max": round(float(non_null.max()), 4),
                "skewness": round(float(non_null.skew()), 4),
                "kurtosis": round(float(non_null.kurtosis()), 4),
                "q25": round(float(non_null.quantile(0.25)), 4),
                "q75": round(float(non_null.quantile(0.75)), 4),
            }
            # Detect if it looks like a binary flag
            if set(non_null.unique()).issubset({0, 1, 0.0, 1.0}):
                result["inferred_type"] = "binary_flag"

    elif pd.api.types.is_datetime64_any_dtype(series):
        result["inferred_type"] = "datetime"
        non_null = series.dropna()
        if len(non_null) > 0:
            result["stats"] = {
                "min_date": str(non_null.min()),
                "max_date": str(non_null.max()),
                "date_range_days": (non_null.max() - non_null.min()).days
            }
    else:
        # Try to parse as datetime
        if series.dtype == object:
            sample = series.dropna().head(20).astype(str)
            date_like = sum(1 for v in sample if _looks_like_date(v))
            if date_like / max(len(sample), 1) > 0.6:
                result["inferred_type"] = "datetime_string"
            else:
                result["inferred_type"] = "categorical" if series.nunique() <= 50 else "text"
                if result["inferred_type"] == "categorical":
                    top_vals = series.value_counts().head(10)
                    result["top_values"] = top_vals.to_dict()

    return result


def _looks_like_date(value: str) -> bool:
    """Quick regex check for date-like strings."""
    patterns = [
        r'\d{4}-\d{2}-\d{2}',  # ISO
        r'\d{2}/\d{2}/\d{4}',  # US
        r'\d{2}-\d{2}-\d{4}',  # EU
        r'\d{4}/\d{2}/\d{2}',  # Alt ISO
    ]
    return any(re.search(p, value) for p in patterns)


def detect_dataset_type(df: pd.DataFrame, schema: dict) -> str:
    """Classify the dataset as tabular, time-series, transactional, etc."""
    temporal_cols = [col for col, info in schema.items() if info["role"] in ("temporal", "datetime_string")]
    id_cols = [col for col, info in schema.items() if info["role"] == "identifier"]

    if len(temporal_cols) >= 1 and len(df) > 100:
        if len(id_cols) >= 2:
            return "transactional"
        return "time_series"
    if len(id_cols) >= 2:
        return "relational"
    return "tabular"


def detect_leakage_risks(df: pd.DataFrame, schema: dict, intent: str) -> list:
    """Detect potential data leakage columns for ML use cases."""
    risks = []
    for col, info in schema.items():
        if info["role"] == "identifier":
            risks.append({
                "column": col,
                "risk": "identifier_leakage",
                "severity": "medium",
                "message": f"Column '{col}' appears to be an ID/key column. Should be dropped for ML training."
            })
        if LEAKAGE_PATTERNS.search(col) and info["role"] == "potential_target":
            risks.append({
                "column": col,
                "risk": "target_leakage",
                "severity": "high",
                "message": f"Column '{col}' may contain future information — possible data leakage."
            })
    return risks


class SchemaEngine:
    def __init__(self):
        pass

    def analyze(self, df: pd.DataFrame, intent: str = "general_ml") -> dict:
        """Full schema analysis of a dataframe."""
        schema = {}
        for col in df.columns:
            col_type = detect_column_type(df[col])
            role = infer_column_role(col, df[col])
            schema[col] = {**col_type, "role": role}

        dataset_type = detect_dataset_type(df, schema)
        leakage_risks = detect_leakage_risks(df, schema, intent)

        # Overall dataset health
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = int(df.duplicated().sum())

        # Identify potential target columns
        potential_targets = [col for col, info in schema.items()
                             if info["role"] == "potential_target"]

        return {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "dataset_type": dataset_type,
            "columns": schema,
            "potential_targets": potential_targets,
            "leakage_risks": leakage_risks,
            "summary": {
                "total_cells": int(total_cells),
                "missing_cells": int(missing_cells),
                "missing_pct": round(missing_cells / max(total_cells, 1) * 100, 2),
                "duplicate_rows": duplicate_rows,
                "duplicate_pct": round(duplicate_rows / max(df.shape[0], 1) * 100, 2),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            }
        }
