"""
DataForge AI — FastAPI Backend
Intelligent ML Preprocessing Platform
"""
import os, json, traceback, io
from datetime import datetime
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from engines import (IntentClassifier, SchemaEngine, NoiseEngine,
                     FeatureEngine, AnomalyEngine, RecommendationEngine, ReportingEngine)
from utils import load_dataframe, df_to_records_safe, sanitize_for_json, validate_prompt, get_file_info

app = FastAPI(title="DataForge AI", description="Context-Aware ML Preprocessing Platform", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

print("[DataForge] Initializing engines...")
intent_clf  = IntentClassifier()
schema_eng  = SchemaEngine()
noise_eng   = NoiseEngine()
feat_eng    = FeatureEngine()
anomaly_eng = AnomalyEngine()
rec_eng     = RecommendationEngine()
report_eng  = ReportingEngine()
print("[DataForge] All engines ready.\n")

@app.get("/")
def root():
    return {"service":"DataForge AI","version":"1.0.0","status":"running","docs":"/docs"}

@app.get("/health")
def health():
    return {"status":"ok","timestamp":datetime.utcnow().isoformat()+"Z"}

@app.post("/api/recommend")
async def recommend_datasets(prompt: str = Form(...)):
    try:
        prompt = validate_prompt(prompt)
        intent_result = intent_clf.classify(prompt)
        intent = intent_result["intent"]
        return JSONResponse(sanitize_for_json({
            "success": True, "prompt": prompt,
            "intent": intent_result,
            "policy": intent_clf.get_cleaning_policy(intent),
            "recommendations": rec_eng.recommend(prompt, intent, top_k=4),
            "learning_path": rec_eng.get_learning_path(intent),
        }))
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, str(e))

@app.post("/api/analyze")
async def analyze_dataset(file: UploadFile = File(...),
                           prompt: str = Form(default="clean this dataset for machine learning")):
    try:
        prompt = validate_prompt(prompt)
        df = load_dataframe(file)
        intent_result = intent_clf.classify(prompt)
        intent = intent_result["intent"]
        schema = schema_eng.analyze(df, intent)
        miss = df.isnull().mean().mean() * 100
        dup  = df.duplicated().mean() * 100
        quality = round((100 - miss) * 0.6 + (100 - dup) * 0.4, 2)
        return JSONResponse(sanitize_for_json({
            "success": True, "file": get_file_info(file),
            "intent": intent_result, "schema": schema,
            "quality_score": quality,
            "sample": df_to_records_safe(df, limit=5),
            "columns": list(df.columns),
        }))
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, str(e))

@app.post("/api/process")
async def process_dataset(
    file: UploadFile = File(...),
    prompt: str = Form(default="clean this dataset for machine learning"),
    run_anomaly_detection: bool = Form(default=True),
    run_feature_engineering: bool = Form(default=True),
):
    try:
        prompt = validate_prompt(prompt)
        df_orig = load_dataframe(file)
        intent_result = intent_clf.classify(prompt)
        intent = intent_result["intent"]
        policy = intent_clf.get_cleaning_policy(intent)
        schema = schema_eng.analyze(df_orig, intent)

        df_clean, transform_log = noise_eng.clean(df_orig, schema["columns"], intent, policy)
        quality_scores = noise_eng.compute_quality_score(df_orig, df_clean, transform_log)

        anomaly_result = {"anomaly_count":0,"suspected_count":0,"anomaly_details":[],
                          "method":"skipped","preserved":policy.get("preserve_outliers",False),
                          "note":"Anomaly detection disabled."}
        if run_anomaly_detection:
            anomaly_result = anomaly_eng.detect(df_clean, preserve_outliers=policy.get("preserve_outliers",False))

        df_engineered = df_clean.copy()
        feature_log = []
        if run_feature_engineering:
            cs = schema_eng.analyze(df_clean, intent)
            df_engineered, feature_log = feat_eng.engineer(df_clean, cs["columns"], intent, policy)

        targets = schema.get("potential_targets", [])
        feat_import = feat_eng.detect_feature_importance(df_engineered, targets[0] if targets else None)
        recommendations = rec_eng.recommend(prompt, intent, top_k=3)
        report = report_eng.generate(
            df_original=df_orig, df_cleaned=df_clean, df_engineered=df_engineered,
            schema=schema["columns"], intent=intent, policy=policy,
            transformation_log=transform_log.to_list(), feature_log=feature_log,
            anomaly_result=anomaly_result, quality_scores=quality_scores,
        )

        return JSONResponse(sanitize_for_json({
            "success": True, "file": get_file_info(file),
            "intent": intent_result,
            "policy": {"name":intent,"description":policy["description"],
                       "preserve_outliers":policy.get("preserve_outliers",False),
                       "risk_notes":policy.get("risk_notes",[])},
            "schema": schema, "quality_scores": quality_scores,
            "anomaly_detection": anomaly_result,
            "feature_importance": feat_import[:15],
            "recommendations": recommendations, "report": report,
            "shape": {"original":list(df_orig.shape),"cleaned":list(df_clean.shape),"engineered":list(df_engineered.shape)},
            "preview": {"original":df_to_records_safe(df_orig,8),"cleaned":df_to_records_safe(df_clean,8),"engineered":df_to_records_safe(df_engineered,8)},
        }))
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, f"Processing failed: {str(e)}")

@app.post("/api/download/cleaned")
async def download_cleaned(file: UploadFile=File(...), prompt: str=Form(default="clean this dataset"),
                            format: str=Form(default="csv")):
    try:
        prompt = validate_prompt(prompt)
        df_orig = load_dataframe(file)
        intent = intent_clf.classify(prompt)["intent"]
        policy = intent_clf.get_cleaning_policy(intent)
        schema = schema_eng.analyze(df_orig, intent)
        df_clean, _ = noise_eng.clean(df_orig, schema["columns"], intent, policy)
        cs = schema_eng.analyze(df_clean, intent)
        df_eng, _ = feat_eng.engineer(df_clean, cs["columns"], intent, policy)
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        stem = (file.filename or "data").rsplit(".", 1)[0]
        if format == "csv":
            buf = io.StringIO()
            df_eng.to_csv(buf, index=False); buf.seek(0)
            return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                     headers={"Content-Disposition": f'attachment; filename="{stem}_cleaned.csv"'})
        elif format == "excel":
            buf = io.BytesIO()
            df_eng.to_excel(buf, index=False); buf.seek(0)
            return StreamingResponse(iter([buf.getvalue()]),
                                     media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                     headers={"Content-Disposition": f'attachment; filename="{stem}_cleaned.xlsx"'})
        else:
            buf = io.BytesIO()
            df_eng.to_parquet(buf, index=False); buf.seek(0)
            return StreamingResponse(iter([buf.getvalue()]), media_type="application/octet-stream",
                                     headers={"Content-Disposition": f'attachment; filename="{stem}_cleaned.parquet"'})
    except HTTPException: raise
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, str(e))

@app.get("/api/model-stats")
def model_stats():
    p = os.path.join(os.path.dirname(__file__), "..", "ml", "saved_models", "training_report.json")
    if os.path.exists(p):
        with open(p) as f: return JSONResponse({"success":True,"stats":json.load(f)})
    return JSONResponse({"success":False,"stats":{}})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

@app.post("/api/search-datasets")
async def search_datasets_endpoint(
    prompt: str = Form(...),
    intent: str = Form(default="general_ml"),
):
    """Search Kaggle + HuggingFace + curated metadata for real datasets."""
    try:
        from engines.kaggle_search import search_datasets
        prompt = validate_prompt(prompt)
        results = search_datasets(prompt, intent, limit=8)
        return JSONResponse(sanitize_for_json({"success": True, "results": results, "total": len(results)}))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
