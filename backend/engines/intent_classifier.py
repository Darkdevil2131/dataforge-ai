"""
Intent Classification Engine
Uses TF-IDF + LogisticRegression trained on synthetic data.
No GPU, no API, runs on free tier.
"""
import json
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


TRAINING_DATA = [
    # ── FRAUD / ANOMALY DETECTION ──────────────────────────────────────────
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

    # ── CLASSIFICATION ─────────────────────────────────────────────────────
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
    ("survival prediction", "classification"),
    ("predict whether customer will buy", "classification"),
    ("churn prediction model", "classification"),
    ("diabetes prediction", "classification"),
    ("cancer classification", "classification"),

    # ── REGRESSION ─────────────────────────────────────────────────────────
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

    # ── FORECASTING ────────────────────────────────────────────────────────
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
    ("ARIMA or LSTM forecasting", "forecasting"),

    # ── ANALYTICS / DASHBOARD ──────────────────────────────────────────────
    ("build a dashboard from this data", "analytics"),
    ("clean and optimize for sales dashboard", "analytics"),
    ("prepare for BI tool", "analytics"),
    ("create KPI report", "analytics"),
    ("visualize business metrics", "analytics"),
    ("prepare for Power BI", "analytics"),
    ("prepare for Tableau", "analytics"),
    ("analytics dashboard", "analytics"),
    ("executive reporting", "analytics"),
    ("business intelligence preparation", "analytics"),
    ("summarize key metrics", "analytics"),
    ("sales performance analysis", "analytics"),
    ("customer behavior analytics", "analytics"),

    # ── RECOMMENDATION ─────────────────────────────────────────────────────
    ("build a recommendation system", "recommendation"),
    ("recommend products to users", "recommendation"),
    ("movie recommendation engine", "recommendation"),
    ("collaborative filtering model", "recommendation"),
    ("product recommendation", "recommendation"),
    ("user-item matrix", "recommendation"),
    ("content-based filtering", "recommendation"),
    ("suggest similar items", "recommendation"),

    # ── NLP / TEXT ─────────────────────────────────────────────────────────
    ("sentiment analysis on reviews", "nlp"),
    ("text classification task", "nlp"),
    ("natural language processing", "nlp"),
    ("topic modeling", "nlp"),
    ("named entity recognition", "nlp"),
    ("text preprocessing for NLP", "nlp"),
    ("clean text data", "nlp"),
    ("process customer reviews", "nlp"),
    ("tweet sentiment analysis", "nlp"),
    ("document classification", "nlp"),

    # ── GENERAL ML ────────────────────────────────────────────────────────
    ("I want to learn machine learning", "general_ml"),
    ("beginner ML project", "general_ml"),
    ("general machine learning dataset", "general_ml"),
    ("practice ML skills", "general_ml"),
    ("explore this dataset", "general_ml"),
    ("understand the data", "general_ml"),
    ("clean this dataset", "general_ml"),
    ("prepare data for machine learning", "general_ml"),
    ("I want to build my first ML model", "general_ml"),
    ("analyze this CSV file", "general_ml"),
]


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "intent_classifier.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl")


def train_and_save():
    """Train the intent classifier and save to disk."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]

    le = LabelEncoder()
    y = le.fit_transform(labels)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
            min_df=1
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ))
    ])

    pipeline.fit(texts, y)

    scores = cross_val_score(pipeline, texts, y, cv=3, scoring="accuracy")
    print(f"[IntentClassifier] CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"[IntentClassifier] Saved to {MODEL_PATH}")
    return pipeline, le


def load_or_train():
    """Load existing model or train if not found."""
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        pipeline = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
    else:
        pipeline, le = train_and_save()
    return pipeline, le


class IntentClassifier:
    def __init__(self):
        self.pipeline, self.le = load_or_train()

    def classify(self, prompt: str) -> dict:
        """
        Classify user intent from natural language prompt.
        Returns intent label + probabilities.
        """
        prompt = prompt.strip().lower()
        if not prompt:
            return {"intent": "general_ml", "confidence": 0.5, "all_scores": {}}

        proba = self.pipeline.predict_proba([prompt])[0]
        classes = self.le.classes_

        scores = {cls: float(prob) for cls, prob in zip(classes, proba)}
        top_intent = max(scores, key=scores.get)
        confidence = scores[top_intent]

        return {
            "intent": top_intent,
            "confidence": round(confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
        }

    def get_cleaning_policy(self, intent: str) -> dict:
        """
        Return data cleaning policy based on detected intent.
        This is the POLICY ENGINE — core of accuracy.
        """
        policies = {
            "fraud_detection": {
                "preserve_outliers": True,
                "aggressive_cleaning": False,
                "balance_classes": False,
                "normalize_skew": False,
                "description": "Preserve anomalies and rare patterns. Fraud data naturally contains outliers that are signals, not noise.",
                "risk_notes": ["Do NOT remove high-value transactions", "Do NOT normalize away rare patterns", "Flag suspicious entries instead of deleting"]
            },
            "classification": {
                "preserve_outliers": False,
                "aggressive_cleaning": True,
                "balance_classes": True,
                "normalize_skew": True,
                "description": "Standard cleaning with class balancing. Outliers handled by IQR/z-score.",
                "risk_notes": ["Check class imbalance", "Encode all categoricals", "Handle missing values carefully"]
            },
            "regression": {
                "preserve_outliers": False,
                "aggressive_cleaning": True,
                "balance_classes": False,
                "normalize_skew": True,
                "description": "Strong normalization. Outlier handling critical for regression targets.",
                "risk_notes": ["Log-transform skewed targets", "Remove extreme outliers from target variable", "Standardize feature scales"]
            },
            "forecasting": {
                "preserve_outliers": True,
                "aggressive_cleaning": False,
                "balance_classes": False,
                "normalize_skew": False,
                "description": "Preserve chronological order. Never shuffle. Outliers may be real events.",
                "risk_notes": ["NEVER shuffle time-series data", "Preserve timestamps", "Lag features will be created", "Rolling windows applied"]
            },
            "analytics": {
                "preserve_outliers": False,
                "aggressive_cleaning": True,
                "balance_classes": False,
                "normalize_skew": False,
                "description": "Focus on aggregations, KPIs, groupings. Remove noise for clear visuals.",
                "risk_notes": ["Create date dimensions", "Aggregate metrics", "Star schema output", "Null handling important for BI"]
            },
            "recommendation": {
                "preserve_outliers": True,
                "aggressive_cleaning": False,
                "balance_classes": False,
                "normalize_skew": False,
                "description": "Preserve user-item relationships. Do not remove sparse interactions.",
                "risk_notes": ["Preserve user IDs", "Preserve item IDs", "Handle cold-start users carefully"]
            },
            "nlp": {
                "preserve_outliers": False,
                "aggressive_cleaning": True,
                "balance_classes": True,
                "normalize_skew": False,
                "description": "Text normalization pipeline. Remove noise from text columns.",
                "risk_notes": ["Normalize text encoding", "Handle special characters", "Check language consistency"]
            },
            "general_ml": {
                "preserve_outliers": False,
                "aggressive_cleaning": True,
                "balance_classes": True,
                "normalize_skew": True,
                "description": "Standard preprocessing pipeline suitable for general ML tasks.",
                "risk_notes": ["Apply standard scaling", "Handle missing values with median/mode", "Encode categoricals"]
            }
        }
        return policies.get(intent, policies["general_ml"])
