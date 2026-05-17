"""
train_intent_classifier.py
==========================
Full training pipeline for the Intent Classification model.
Trains on synthetic + augmented data. Saves model to saved_models/.
Run: python ml/train_intent_classifier.py

No GPU needed. Runs on CPU in < 30 seconds.
"""
import os
import sys
import json
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(ROOT, "ml", "saved_models")
BACKEND_MODEL_DIR = os.path.join(ROOT, "backend", "models")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BACKEND_MODEL_DIR, exist_ok=True)

# ── Full training dataset (400+ examples across 8 classes) ───────────────────
TRAINING_DATA = [

    # ── FRAUD DETECTION ──────────────────────────────────────────────────────
    ("detect fraud in transactions", "fraud_detection"),
    ("prepare for fraud detection model", "fraud_detection"),
    ("identify fraudulent activity", "fraud_detection"),
    ("flag suspicious transactions", "fraud_detection"),
    ("find anomalies in payment data", "fraud_detection"),
    ("credit card fraud detection", "fraud_detection"),
    ("anomaly detection in financial data", "fraud_detection"),
    ("identify unusual patterns in transactions", "fraud_detection"),
    ("build a fraud classifier", "fraud_detection"),
    ("detect outliers in banking data", "fraud_detection"),
    ("suspicious behavior detection", "fraud_detection"),
    ("money laundering detection", "fraud_detection"),
    ("fraudulent transaction identification", "fraud_detection"),
    ("payment fraud model", "fraud_detection"),
    ("insurance fraud detection", "fraud_detection"),
    ("bank fraud classification", "fraud_detection"),
    ("abnormal activity in accounts", "fraud_detection"),
    ("detect fake accounts", "fraud_detection"),
    ("identity theft detection", "fraud_detection"),
    ("transaction anomaly scoring", "fraud_detection"),
    ("ecommerce fraud prevention model", "fraud_detection"),
    ("financial crime detection", "fraud_detection"),
    ("card not present fraud", "fraud_detection"),
    ("detect unauthorized transactions", "fraud_detection"),
    ("anti-fraud ML model", "fraud_detection"),

    # ── CLASSIFICATION ───────────────────────────────────────────────────────
    ("build a classification model", "classification"),
    ("predict customer churn", "classification"),
    ("binary classification task", "classification"),
    ("multiclass classification", "classification"),
    ("predict disease outcome", "classification"),
    ("categorize customer complaints", "classification"),
    ("classify emails as spam or not", "classification"),
    ("predict loan default", "classification"),
    ("customer segmentation model", "classification"),
    ("predict heart disease", "classification"),
    ("survival prediction titanic", "classification"),
    ("predict whether customer will buy", "classification"),
    ("churn prediction model", "classification"),
    ("diabetes prediction", "classification"),
    ("cancer classification model", "classification"),
    ("sentiment classification positive negative", "classification"),
    ("predict employee attrition", "classification"),
    ("product defect classification", "classification"),
    ("credit risk classification", "classification"),
    ("predict hospital readmission", "classification"),
    ("medical diagnosis model", "classification"),
    ("spam ham email classifier", "classification"),
    ("predict customer purchase intent", "classification"),
    ("default risk binary classifier", "classification"),
    ("multi-label classification problem", "classification"),

    # ── REGRESSION ───────────────────────────────────────────────────────────
    ("predict house prices", "regression"),
    ("estimate sales revenue", "regression"),
    ("predict salary based on experience", "regression"),
    ("forecast demand for products", "regression"),
    ("regression model for continuous output", "regression"),
    ("predict stock price value", "regression"),
    ("property valuation model", "regression"),
    ("customer lifetime value prediction", "regression"),
    ("estimate insurance cost", "regression"),
    ("price prediction model", "regression"),
    ("predict energy consumption", "regression"),
    ("real estate price estimation", "regression"),
    ("predict car price from features", "regression"),
    ("predict employee salary", "regression"),
    ("continuous output prediction", "regression"),
    ("hospital cost estimation", "regression"),
    ("revenue prediction model", "regression"),
    ("predict age from features", "regression"),
    ("predict delivery time regression", "regression"),
    ("forecast product price over time", "regression"),

    # ── FORECASTING ──────────────────────────────────────────────────────────
    ("time series forecasting", "forecasting"),
    ("forecast sales next quarter", "forecasting"),
    ("predict future demand", "forecasting"),
    ("make this usable for time series forecasting", "forecasting"),
    ("stock price prediction over time", "forecasting"),
    ("energy consumption forecasting", "forecasting"),
    ("forecast website traffic", "forecasting"),
    ("revenue forecasting model", "forecasting"),
    ("predict next month sales", "forecasting"),
    ("weather forecasting dataset", "forecasting"),
    ("seasonal trend analysis", "forecasting"),
    ("ARIMA LSTM forecasting model", "forecasting"),
    ("predict future inventory levels", "forecasting"),
    ("supply chain demand forecasting", "forecasting"),
    ("retail sales prediction time series", "forecasting"),
    ("prophet forecasting seasonal data", "forecasting"),
    ("long short term memory forecasting", "forecasting"),
    ("time dependent prediction model", "forecasting"),
    ("multi-step ahead forecasting", "forecasting"),
    ("financial time series prediction", "forecasting"),

    # ── ANALYTICS ────────────────────────────────────────────────────────────
    ("build a dashboard from this data", "analytics"),
    ("clean and optimize for sales dashboard", "analytics"),
    ("prepare for BI tool", "analytics"),
    ("create KPI report", "analytics"),
    ("visualize business metrics", "analytics"),
    ("prepare for Power BI", "analytics"),
    ("prepare for Tableau", "analytics"),
    ("analytics dashboard", "analytics"),
    ("executive reporting dataset", "analytics"),
    ("business intelligence preparation", "analytics"),
    ("summarize key metrics for report", "analytics"),
    ("sales performance analysis", "analytics"),
    ("customer behavior analytics", "analytics"),
    ("monthly revenue dashboard", "analytics"),
    ("create star schema for BI", "analytics"),
    ("RFM customer analysis", "analytics"),
    ("cohort analysis dashboard", "analytics"),
    ("funnel analysis dataset", "analytics"),
    ("marketing analytics report", "analytics"),
    ("prepare for Apache Superset", "analytics"),

    # ── RECOMMENDATION ───────────────────────────────────────────────────────
    ("build a recommendation system", "recommendation"),
    ("recommend products to users", "recommendation"),
    ("movie recommendation engine", "recommendation"),
    ("collaborative filtering model", "recommendation"),
    ("product recommendation", "recommendation"),
    ("user-item matrix collaborative", "recommendation"),
    ("content-based filtering model", "recommendation"),
    ("suggest similar items to users", "recommendation"),
    ("build Netflix-like recommender", "recommendation"),
    ("Amazon product recommendation", "recommendation"),
    ("playlist recommendation model", "recommendation"),
    ("restaurant recommendation engine", "recommendation"),
    ("user preference model", "recommendation"),
    ("item-based collaborative filtering", "recommendation"),
    ("matrix factorization SVD recommendation", "recommendation"),

    # ── NLP ──────────────────────────────────────────────────────────────────
    ("sentiment analysis on reviews", "nlp"),
    ("text classification task", "nlp"),
    ("natural language processing", "nlp"),
    ("topic modeling LDA", "nlp"),
    ("named entity recognition NER", "nlp"),
    ("text preprocessing for NLP model", "nlp"),
    ("clean text data for BERT", "nlp"),
    ("process customer reviews NLP", "nlp"),
    ("tweet sentiment analysis", "nlp"),
    ("document classification NLP", "nlp"),
    ("text summarization model", "nlp"),
    ("question answering NLP", "nlp"),
    ("chatbot training data preparation", "nlp"),
    ("text mining dataset preparation", "nlp"),
    ("language model fine-tuning data", "nlp"),

    # ── GENERAL ML ───────────────────────────────────────────────────────────
    ("I want to learn machine learning", "general_ml"),
    ("beginner ML project dataset", "general_ml"),
    ("general machine learning dataset", "general_ml"),
    ("practice ML skills with dataset", "general_ml"),
    ("explore this dataset EDA", "general_ml"),
    ("understand the data distribution", "general_ml"),
    ("clean this dataset for modeling", "general_ml"),
    ("prepare data for machine learning", "general_ml"),
    ("I want to build my first ML model", "general_ml"),
    ("analyze this CSV file", "general_ml"),
    ("data exploration and preprocessing", "general_ml"),
    ("feature engineering for ML", "general_ml"),
    ("baseline machine learning model", "general_ml"),
    ("end to end ML pipeline", "general_ml"),
    ("unsupervised learning clustering", "general_ml"),
]


def build_augmented_corpus(data):
    """Add light augmentation: lowercase variants, synonym swaps."""
    augmented = list(data)
    synonyms = {
        "predict": "estimate",
        "build": "create",
        "detect": "identify",
        "model": "classifier",
        "dataset": "data",
        "prepare": "get ready",
        "forecast": "project",
    }
    for text, label in data:
        for word, synonym in synonyms.items():
            if word in text.lower():
                new_text = text.lower().replace(word, synonym)
                augmented.append((new_text, label))
                break  # one augmentation per sample
    return augmented


def train():
    print("=" * 60)
    print("  DataForge AI — Intent Classifier Training")
    print("=" * 60)

    data = build_augmented_corpus(TRAINING_DATA)
    texts = [d[0] for d in data]
    labels = [d[1] for d in data]

    print(f"\n  Total training samples: {len(texts)}")
    print(f"  Classes: {sorted(set(labels))}\n")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # ── Model: TF-IDF + Logistic Regression (best for short text) ────────────
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
            analyzer="word",
        )),
        ("clf", LogisticRegression(
            C=3.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ))
    ])

    # ── Cross-validation ─────────────────────────────────────────────────────
    print("  Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr_pipeline, texts, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(lr_pipeline, texts, y, cv=cv, scoring="f1_weighted")

    print(f"  CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  CV F1-Score : {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # ── Final train + test split report ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Retrain on full data ──────────────────────────────────────────────────
    lr_pipeline.fit(texts, y)

    # ── Save models ───────────────────────────────────────────────────────────
    model_path = os.path.join(SAVE_DIR, "intent_classifier.pkl")
    encoder_path = os.path.join(SAVE_DIR, "label_encoder.pkl")
    backend_model = os.path.join(BACKEND_MODEL_DIR, "intent_classifier.pkl")
    backend_encoder = os.path.join(BACKEND_MODEL_DIR, "label_encoder.pkl")

    joblib.dump(lr_pipeline, model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(lr_pipeline, backend_model)
    joblib.dump(le, backend_encoder)

    # ── Save training metadata ────────────────────────────────────────────────
    meta = {
        "model": "TF-IDF + LogisticRegression",
        "n_samples": len(texts),
        "classes": list(le.classes_),
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "cv_f1": round(float(f1_scores.mean()), 4),
        "ngram_range": [1, 3],
        "max_features": 8000,
    }
    with open(os.path.join(SAVE_DIR, "intent_model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Model saved → ml/saved_models/intent_classifier.pkl")
    print(f"  ✅ Model synced → backend/models/intent_classifier.pkl")
    print(f"  CV Accuracy: {cv_scores.mean():.1%}")
    print("=" * 60)

    # ── Quick inference test ──────────────────────────────────────────────────
    test_prompts = [
        "detect fraud in credit card transactions",
        "predict house prices regression",
        "time series forecasting for sales",
        "I want to learn machine learning",
        "prepare for BI dashboard Power BI",
    ]
    print("\n  Inference test:")
    for prompt in test_prompts:
        proba = lr_pipeline.predict_proba([prompt])[0]
        top_idx = proba.argmax()
        print(f"    '{prompt}'")
        print(f"    → {le.classes_[top_idx]} ({proba[top_idx]:.0%})\n")

    return lr_pipeline, le


if __name__ == "__main__":
    train()
