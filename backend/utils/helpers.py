"""
utils/helpers.py
================
Shared utilities for the DataForge AI backend.
Robust universal dataset loading + dataframe safety.
"""

import io
import json
import csv
import numpy as np
import pandas as pd
import chardet

from pathlib import Path
from fastapi import HTTPException
from typing import Any

MAX_ROWS = 100_000


def detect_encoding(content: bytes):
    """Detect file encoding safely."""
    result = chardet.detect(content)
    return result["encoding"] or "utf-8"


def load_dataframe(file) -> pd.DataFrame:
    """
    Universal dataset loader.

    Supports:
    - CSV
    - malformed CSV
    - Excel (.xlsx/.xls)
    - JSON
    - Parquet

    Auto detects:
    - encoding
    - delimiters
    """

    content = file.file.read()
    fname = (file.filename or "").lower()

    try:

        # =====================================================
        # CSV
        # =====================================================
        if fname.endswith(".csv"):

            encoding = detect_encoding(content)

            try:
                # Try auto delimiter detection
                sample = content[:5000].decode(encoding, errors="ignore")

                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter

                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    sep=delimiter,
                    on_bad_lines="skip",
                    engine="python"
                )

            except Exception:
                # Fallback safe parser
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    on_bad_lines="skip",
                    engine="python"
                )

        # =====================================================
        # EXCEL
        # =====================================================
        elif fname.endswith((".xlsx", ".xls")):

            df = pd.read_excel(io.BytesIO(content))

        # =====================================================
        # JSON
        # =====================================================
        elif fname.endswith(".json"):

            try:
                df = pd.read_json(io.BytesIO(content))

            except Exception:
                raw = json.loads(content.decode("utf-8"))
                df = pd.json_normalize(raw)

        # =====================================================
        # PARQUET
        # =====================================================
        elif fname.endswith(".parquet"):

            df = pd.read_parquet(io.BytesIO(content))

        # =====================================================
        # FALLBACK AUTO DETECT
        # =====================================================
        else:

            encoding = detect_encoding(content)

            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    on_bad_lines="skip",
                    engine="python"
                )

            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unsupported file format. "
                        "Upload CSV, Excel (.xlsx), JSON, or Parquet."
                    )
                )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse file: {str(e)}"
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(df) < 2:
        raise HTTPException(
            status_code=400,
            detail="Dataset has fewer than 2 rows."
        )

    # Free tier safety
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)

    # Remove fully empty columns
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = [
        str(col).strip().replace(" ", "_")
        for col in df.columns
    ]

    return df


def df_to_records_safe(df: pd.DataFrame, limit: int = 100) -> list:
    """
    Convert DataFrame to JSON-safe records.
    """

    sample = df.head(limit).copy()

    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].astype(str)

    sample = sample.replace([np.nan, np.inf, -np.inf], None)

    records = []

    for row in sample.to_dict(orient="records"):

        clean_row = {}

        for k, v in row.items():

            if isinstance(v, np.integer):
                clean_row[k] = int(v)

            elif isinstance(v, np.floating):
                clean_row[k] = None if np.isnan(v) else float(v)

            elif isinstance(v, np.bool_):
                clean_row[k] = bool(v)

            else:
                clean_row[k] = v

        records.append(clean_row)

    return records


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python.
    """

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
    """
    Validate user prompt.
    """

    if not prompt or not prompt.strip():
        return "clean this dataset for machine learning"

    prompt = prompt.strip()

    if len(prompt) > 500:
        prompt = prompt[:500]

    return prompt


def get_file_info(file) -> dict:
    """
    Extract uploaded file metadata.
    """

    return {
        "filename": file.filename or "unknown",
        "content_type": file.content_type or "unknown",
        "extension": (
            (file.filename or "").rsplit(".", 1)[-1].lower()
            if "." in (file.filename or "")
            else "unknown"
        ),
    }