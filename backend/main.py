"""
DataForge AI — FastAPI Backend
Optimized Version
"""

import os
import json
import traceback
import io
from datetime import datetime

import numpy as np
import pandas as pd
import uvicorn

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
)

from engines import (
    IntentClassifier,
    SchemaEngine,
    NoiseEngine,
    FeatureEngine,
    AnomalyEngine,
    RecommendationEngine,
    ReportingEngine,
)

from utils import (
    load_dataframe,
    df_to_records_safe,
    sanitize_for_json,
    validate_prompt,
    get_file_info,
)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="DataForge AI",
    description="Context-Aware ML Preprocessing Platform",
    version="2.0.0",
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# INITIALIZE ENGINES
# =========================================================

print("[DataForge] Initializing engines...")

intent_clf = IntentClassifier()

schema_eng = SchemaEngine()

noise_eng = NoiseEngine()

feat_eng = FeatureEngine()

anomaly_eng = AnomalyEngine()

rec_eng = RecommendationEngine()

report_eng = ReportingEngine()

print("[DataForge] All engines ready.")

# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "service": "DataForge AI",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

# =========================================================
# DATASET RECOMMENDATION
# =========================================================

@app.post("/api/recommend")
async def recommend_datasets(
    prompt: str = Form(...),
):
    try:
        prompt = validate_prompt(prompt)

        intent_result = intent_clf.classify(prompt)

        intent = intent_result["intent"]

        return JSONResponse(
            sanitize_for_json(
                {
                    "success": True,
                    "prompt": prompt,
                    "intent": intent_result,
                    "policy": intent_clf.get_cleaning_policy(intent),
                    "recommendations": rec_eng.recommend(
                        prompt,
                        intent,
                        top_k=4,
                    ),
                    "learning_path": rec_eng.get_learning_path(
                        intent
                    ),
                }
            )
        )

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

# =========================================================
# ANALYZE DATASET
# =========================================================

@app.post("/api/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    prompt: str = Form(
        default="clean this dataset for machine learning"
    ),
):
    try:
        prompt = validate_prompt(prompt)

        print("[DataForge] Loading dataset...")

        df = load_dataframe(file)

        # =========================================
        # HUGE DATASET FIX
        # =========================================

        if len(df) > 10000:
            df = df.head(10000)

        intent_result = intent_clf.classify(prompt)

        intent = intent_result["intent"]

        schema = schema_eng.analyze(df, intent)

        missing_percent = (
            df.isnull().mean().mean() * 100
        )

        duplicate_percent = (
            df.duplicated().mean() * 100
        )

        quality_score = round(
            ((100 - missing_percent) * 0.6)
            + ((100 - duplicate_percent) * 0.4),
            2,
        )

        return JSONResponse(
            sanitize_for_json(
                {
                    "success": True,
                    "file": get_file_info(file),
                    "intent": intent_result,
                    "schema": schema,
                    "quality_score": quality_score,
                    "sample": df_to_records_safe(df, limit=3),
                    "columns": list(df.columns),
                }
            )
        )

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

# =========================================================
# PROCESS DATASET
# =========================================================

@app.post("/api/process")
async def process_dataset(
    file: UploadFile = File(...),
    prompt: str = Form(
        default="clean this dataset for machine learning"
    ),
    run_anomaly_detection: bool = Form(default=True),
    run_feature_engineering: bool = Form(default=True),
):
    try:
        print("\n==============================")
        print("[DataForge] PROCESS STARTED")

        prompt = validate_prompt(prompt)

        # =====================================================
        # LOAD DATA
        # =====================================================

        print("[DataForge] Loading dataset...")

        df_original = load_dataframe(file)

        # =====================================================
        # HUGE DATASET FIX
        # =====================================================

        MAX_ROWS = 10000

        if len(df_original) > MAX_ROWS:
            print(
                f"[DataForge] Large dataset detected. Limiting to {MAX_ROWS} rows."
            )

            df_original = df_original.head(MAX_ROWS)

        print(
            f"[DataForge] Dataset shape: {df_original.shape}"
        )

        # =====================================================
        # INTENT
        # =====================================================

        print("[DataForge] Detecting intent...")

        intent_result = intent_clf.classify(prompt)

        intent = intent_result["intent"]

        # =====================================================
        # POLICY
        # =====================================================

        policy = intent_clf.get_cleaning_policy(intent)

        # =====================================================
        # SCHEMA
        # =====================================================

        print("[DataForge] Analyzing schema...")

        schema = schema_eng.analyze(
            df_original,
            intent,
        )

        # =====================================================
        # CLEANING
        # =====================================================

        print("[DataForge] Cleaning dataset...")

        df_cleaned, transform_log = noise_eng.clean(
            df_original,
            schema["columns"],
            intent,
            policy,
        )

        # =====================================================
        # QUALITY SCORES
        # =====================================================

        quality_scores = (
            noise_eng.compute_quality_score(
                df_original,
                df_cleaned,
                transform_log,
            )
        )

        # =====================================================
        # ANOMALY DETECTION
        # =====================================================

        anomaly_result = {
            "anomaly_count": 0,
            "suspected_count": 0,
            "anomaly_details": [],
            "method": "skipped",
            "preserved": False,
            "note": "Skipped for performance optimization.",
        }

        if (
            run_anomaly_detection
            and len(df_cleaned) < 5000
        ):
            print(
                "[DataForge] Running anomaly detection..."
            )

            anomaly_result = anomaly_eng.detect(
                df_cleaned,
                preserve_outliers=policy.get(
                    "preserve_outliers",
                    False,
                ),
            )

        # =====================================================
        # FEATURE ENGINEERING
        # =====================================================

        df_engineered = df_cleaned.copy()

        feature_log = []

        if (
            run_feature_engineering
            and len(df_cleaned) < 5000
        ):
            print(
                "[DataForge] Running feature engineering..."
            )

            clean_schema = schema_eng.analyze(
                df_cleaned,
                intent,
            )

            df_engineered, feature_log = (
                feat_eng.engineer(
                    df_cleaned,
                    clean_schema["columns"],
                    intent,
                    policy,
                )
            )

        # =====================================================
        # FEATURE IMPORTANCE DISABLED
        # =====================================================

        feature_importance = []

        # =====================================================
        # RECOMMENDATIONS
        # =====================================================

        print(
            "[DataForge] Generating recommendations..."
        )

        recommendations = rec_eng.recommend(
            prompt,
            intent,
            top_k=3,
        )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        print("[DataForge] Sending response...")

        return JSONResponse(
            sanitize_for_json(
                {
                    "success": True,

                    "file": get_file_info(file),

                    "intent": intent_result,

                    "policy": {
                        "name": intent,
                        "description": policy[
                            "description"
                        ],
                        "preserve_outliers": policy.get(
                            "preserve_outliers",
                            False,
                        ),
                    },

                    "schema": schema,

                    "quality_scores": quality_scores,

                    "anomaly_detection": anomaly_result,

                    "feature_importance": feature_importance,

                    "recommendations": recommendations,

                    "shape": {
                        "original": list(
                            df_original.shape
                        ),
                        "cleaned": list(
                            df_cleaned.shape
                        ),
                        "engineered": list(
                            df_engineered.shape
                        ),
                    },

                    # SMALL PREVIEW ONLY
                    "preview": {
                        "original": df_to_records_safe(
                            df_original,
                            3,
                        ),
                    },
                }
            )
        )

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
        )

# =========================================================
# DOWNLOAD CLEANED DATA
# =========================================================

@app.post("/api/download/cleaned")
async def download_cleaned(
    file: UploadFile = File(...),
    prompt: str = Form(
        default="clean this dataset"
    ),
    format: str = Form(default="csv"),
):
    try:
        prompt = validate_prompt(prompt)

        df_original = load_dataframe(file)

        if len(df_original) > 10000:
            df_original = df_original.head(10000)

        intent = intent_clf.classify(
            prompt
        )["intent"]

        policy = intent_clf.get_cleaning_policy(
            intent
        )

        schema = schema_eng.analyze(
            df_original,
            intent,
        )

        df_cleaned, _ = noise_eng.clean(
            df_original,
            schema["columns"],
            intent,
            policy,
        )

        df_engineered = df_cleaned.copy()

        df_engineered = df_engineered.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        stem = (
            (file.filename or "data")
            .rsplit(".", 1)[0]
        )

        # =====================================================
        # CSV
        # =====================================================

        if format == "csv":
            buffer = io.StringIO()

            df_engineered.to_csv(
                buffer,
                index=False,
            )

            buffer.seek(0)

            return StreamingResponse(
                iter([buffer.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition":
                    f'attachment; filename="{stem}_cleaned.csv"'
                },
            )

        # =====================================================
        # EXCEL
        # =====================================================

        elif format == "excel":
            buffer = io.BytesIO()

            df_engineered.to_excel(
                buffer,
                index=False,
            )

            buffer.seek(0)

            return StreamingResponse(
                iter([buffer.getvalue()]),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition":
                    f'attachment; filename="{stem}_cleaned.xlsx"'
                },
            )

        # =====================================================
        # PARQUET
        # =====================================================

        else:
            buffer = io.BytesIO()

            df_engineered.to_parquet(
                buffer,
                index=False,
            )

            buffer.seek(0)

            return StreamingResponse(
                iter([buffer.getvalue()]),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition":
                    f'attachment; filename="{stem}_cleaned.parquet"'
                },
            )

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

# =========================================================
# MODEL STATS
# =========================================================

@app.get("/api/model-stats")
def model_stats():
    report_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ml",
        "saved_models",
        "training_report.json",
    )

    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return JSONResponse(
                {
                    "success": True,
                    "stats": json.load(f),
                }
            )

    return JSONResponse(
        {
            "success": False,
            "stats": {},
        }
    )

# =========================================================
# SEARCH DATASETS
# =========================================================

@app.post("/api/search-datasets")
async def search_datasets_endpoint(
    prompt: str = Form(...),
    intent: str = Form(default="general_ml"),
):
    try:
        from engines.kaggle_search import (
            search_datasets,
        )

        prompt = validate_prompt(prompt)

        results = search_datasets(
            prompt,
            intent,
            limit=8,
        )

        return JSONResponse(
            sanitize_for_json(
                {
                    "success": True,
                    "results": results,
                    "total": len(results),
                }
            )
        )

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 10000)
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )