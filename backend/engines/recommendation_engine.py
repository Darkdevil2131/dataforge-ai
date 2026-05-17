"""
Dataset Recommendation Engine
Uses TF-IDF cosine similarity on dataset metadata.
No external API. No GPU. Runs instantly.
"""
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any


METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_metadata.json")

DIFFICULTY_INFO = {
    "beginner": {
        "label": "Beginner",
        "color": "green",
        "description": "Clean, small, well-documented. Perfect to learn ML basics."
    },
    "intermediate": {
        "label": "Intermediate",
        "color": "yellow",
        "description": "Requires feature engineering and careful preprocessing."
    },
    "advanced": {
        "label": "Advanced",
        "color": "red",
        "description": "Complex preprocessing, large-scale, domain expertise needed."
    }
}

TASK_DESCRIPTIONS = {
    "fraud_detection": "Fraud & Anomaly Detection",
    "classification": "Binary / Multi-class Classification",
    "regression": "Regression / Continuous Prediction",
    "forecasting": "Time-Series Forecasting",
    "analytics": "Analytics & BI Dashboard",
    "recommendation": "Recommendation Systems",
    "nlp": "Natural Language Processing",
    "general_ml": "General Machine Learning"
}


class RecommendationEngine:
    def __init__(self):
        with open(METADATA_PATH, "r") as f:
            self.datasets = json.load(f)
        self._build_index()

    def _build_index(self):
        """Build TF-IDF index over dataset descriptions + tags."""
        self.corpus = []
        for ds in self.datasets:
            text = (
                f"{ds['name']} "
                f"{ds['description']} "
                f"{' '.join(ds.get('tags', []))} "
                f"{' '.join(ds.get('task_type', []))} "
                f"{ds.get('domain', '')} "
                f"{' '.join(ds.get('recommended_models', []))}"
            )
            self.corpus.append(text.lower())

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def recommend(self, prompt: str, intent: str,
                  top_k: int = 4) -> list:
        """
        Find the top-k most relevant datasets for the user's prompt + intent.
        Returns enriched dataset recommendations.
        """
        # Augment query with intent keywords
        task_label = TASK_DESCRIPTIONS.get(intent, "")
        augmented_query = f"{prompt} {task_label} {intent}".lower()

        query_vec = self.vectorizer.transform([augmented_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            ds = self.datasets[idx].copy()
            ds["relevance_score"] = round(float(scores[idx]), 4)
            ds["difficulty_info"] = DIFFICULTY_INFO.get(ds.get("difficulty", "beginner"), {})
            ds["task_match"] = intent in ds.get("task_type", [])
            results.append(ds)

        return results

    def get_learning_path(self, intent: str) -> dict:
        """Return structured learning recommendations for beginners."""
        learning_paths = {
            "general_ml": {
                "title": "ML Beginner Path",
                "steps": [
                    {"step": 1, "dataset": "Iris", "task": "Your first classification model", "why": "Only 150 rows, perfectly clean, 3 classes"},
                    {"step": 2, "dataset": "Titanic", "task": "Handle missing values + feature engineering", "why": "Teaches real preprocessing — the most common beginner project"},
                    {"step": 3, "dataset": "House Prices", "task": "Advanced feature engineering + regression", "why": "81 features, many missing values, great regression practice"},
                ],
                "recommended_models": [
                    {"model": "Logistic Regression", "why": "Understand linear decision boundaries"},
                    {"model": "Random Forest", "why": "Powerful, handles noise, feature importance"},
                    {"model": "XGBoost", "why": "Industry standard for tabular data"},
                ]
            },
            "fraud_detection": {
                "title": "Fraud Detection Path",
                "steps": [
                    {"step": 1, "dataset": "Credit Card Fraud", "task": "Binary classification on imbalanced data", "why": "The industry benchmark for fraud ML"},
                ],
                "key_techniques": ["SMOTE oversampling", "Class weights", "Precision-Recall focus (not accuracy)", "IsolationForest for unsupervised"],
                "recommended_models": [
                    {"model": "XGBoost", "why": "Best AUC on tabular fraud data"},
                    {"model": "IsolationForest", "why": "Unsupervised — detects novel fraud"},
                    {"model": "LightGBM", "why": "Fast, handles class imbalance well"},
                ]
            },
            "forecasting": {
                "title": "Forecasting Path",
                "steps": [
                    {"step": 1, "dataset": "E-Commerce Sales", "task": "Sales trend analysis + forecast", "why": "Realistic business time-series"},
                    {"step": 2, "dataset": "Stock Prices", "task": "Financial time-series with lag features", "why": "High variance — teaches robust forecasting"},
                ],
                "key_techniques": ["Lag features", "Rolling mean/std", "Seasonal decomposition", "Never shuffle time-series"],
                "recommended_models": [
                    {"model": "Prophet", "why": "Handles seasonality automatically"},
                    {"model": "ARIMA", "why": "Classical — good baseline"},
                    {"model": "XGBoost", "why": "With lag features — often beats LSTM on tabular"},
                ]
            },
            "classification": {
                "title": "Classification Path",
                "steps": [
                    {"step": 1, "dataset": "Titanic", "task": "Binary survival prediction", "why": "Classic, well-documented, community support"},
                    {"step": 2, "dataset": "Customer Churn", "task": "Business classification", "why": "Real-world imbalance + categorical features"},
                    {"step": 3, "dataset": "Heart Disease", "task": "Medical binary classification", "why": "Domain-sensitive, high stakes"},
                ],
                "recommended_models": [
                    {"model": "Random Forest", "why": "Robust, interpretable feature importance"},
                    {"model": "XGBoost", "why": "State of the art for tabular classification"},
                    {"model": "Logistic Regression", "why": "Interpretable baseline — always start here"},
                ]
            },
            "analytics": {
                "title": "Analytics / BI Path",
                "steps": [
                    {"step": 1, "dataset": "E-Commerce Sales", "task": "Revenue dashboard + RFM analysis", "why": "Real sales data, multi-dimension aggregation"},
                ],
                "key_techniques": ["Star schema design", "RFM segmentation", "Cohort analysis", "KPI aggregation"],
                "recommended_tools": ["Power BI", "Tableau", "Metabase", "Apache Superset (free)"]
            }
        }
        return learning_paths.get(intent, learning_paths["general_ml"])
