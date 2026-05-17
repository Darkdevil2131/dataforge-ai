"""
utils/helpers.py
================
Shared utilities for the DataForge AI backend.
File loading, dataframe safety, validation, format detection.
"""
import io
import json
import numpy as np
import pandas as pd
from fastapi import HTTPException
from typing import Any


MAX_ROWS = 100_000


def load_dataframe(file) -> pd.DataFrame:
    """
    Load an uploaded file into a pandas DataFrame.
    Supports: CSV, Excel (.xlsx/.xls), JSON, Parquet.
    Enforces 100K row limit for free-tier memory safety.
    """
    content = file.file.read()
    fname = (file.filename or "").lower()

    try:
        if fname.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="latin-1")

        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))

        elif fname.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))

        elif fname.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))

        else:
            # Try CSV as fallback
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Upload CSV, Excel (.xlsx), JSON, or Parquet."
                )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Dataset has fewer than 2 rows — cannot process.")

    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)

    return df


def df_to_records_safe(df: pd.DataFrame, limit: int = 100) -> list:
    """
    Convert a DataFrame to JSON-serializable records.
    Handles NaN, Inf, datetime, numpy types safely.
    """
    sample = df.head(limit).copy()

    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].astype(str)

    sample = sample.replace([np.nan, np.inf, -np.inf], None)

    # Cast numpy scalar types to native Python
    records = []
    for row in sample.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                clean_row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean_row[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, np.bool_):
                clean_row[k] = bool(v)
            else:
                clean_row[k] = v
        records.append(clean_row)

    return records


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def validate_prompt(prompt: str) -> str:
    """Validate and sanitize the user prompt."""
    if not prompt or not prompt.strip():
        return "clean this dataset for machine learning"
    prompt = prompt.strip()
    if len(prompt) > 500:
        prompt = prompt[:500]
    return prompt


def get_file_info(file) -> dict:
    """Extract metadata about the uploaded file."""
    return {
        "filename": file.filename or "unknown",
        "content_type": file.content_type or "unknown",
        "extension": (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else "unknown",
    }
