"""
train_all_models.py
===================
Trains ALL ML models for DataForge AI on real datasets.
Saves trained models + evaluation reports to ml/saved_models/

Models trained:
  1. Intent Classifier    — TF-IDF + Logistic Regression
  2. Fraud Detector       — XGBoost + IsolationForest ensemble (on credit_card_fraud.csv)
  3. Churn Classifier     — Random Forest + SMOTE (on customer_churn.csv)
  4. Anomaly Detector     — IsolationForest + LOF (on all datasets)
  5. House Price Regressor — GradientBoosting (on house_prices.csv)
  6. Cancer Classifier    — SVM + RandomForest (on breast_cancer.csv)

Run: python ml/train_all_models.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder, LabelBinarizer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    GradientBoostingRegressor, IsolationForest, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "ml", "datasets")
SAVE_DIR  = os.path.join(ROOT, "ml", "saved_models")
BACK_DIR  = os.path.join(ROOT, "backend", "models")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BACK_DIR, exist_ok=True)

RESULTS = {}

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def save(obj, name):
    """Save model to both ml/saved_models and backend/models."""
    joblib.dump(obj, os.path.join(SAVE_DIR, name))
    joblib.dump(obj, os.path.join(BACK_DIR, name))
    print(f"  ✅ Saved: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. INTENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def train_intent_classifier():
    sep("1/6  Intent Classifier  [TF-IDF + Logistic Regression]")

    TRAINING_DATA = [
        # fraud_detection
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
        ("anti-fraud ML model", "fraud_detection"),
        # classification
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
        ("churn prediction model", "classification"),
        ("diabetes prediction", "classification"),
        ("cancer classification model", "classification"),
        ("predict employee attrition", "classification"),
        ("product defect classification", "classification"),
        ("credit risk classification", "classification"),
        ("predict hospital readmission", "classification"),
        ("medical diagnosis model", "classification"),
        ("spam ham email classifier", "classification"),
        ("default risk binary classifier", "classification"),
        # regression
        ("predict house prices", "regression"),
        ("estimate sales revenue", "regression"),
        ("predict salary based on experience", "regression"),
        ("regression model for continuous output", "regression"),
        ("predict stock price value", "regression"),
        ("property valuation model", "regression"),
        ("customer lifetime value prediction", "regression"),
        ("estimate insurance cost", "regression"),
        ("price prediction model", "regression"),
        ("predict energy consumption", "regression"),
        ("real estate price estimation", "regression"),
        ("predict car price from features", "regression"),
        ("continuous output prediction", "regression"),
        ("revenue prediction model", "regression"),
        ("predict delivery time regression", "regression"),
        # forecasting
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
        ("time dependent prediction model", "forecasting"),
        ("multi-step ahead forecasting", "forecasting"),
        # analytics
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
        ("sales performance analysis", "analytics"),
        ("customer behavior analytics", "analytics"),
        ("monthly revenue dashboard", "analytics"),
        ("create star schema for BI", "analytics"),
        ("RFM customer analysis", "analytics"),
        ("cohort analysis dashboard", "analytics"),
        ("marketing analytics report", "analytics"),
        # recommendation
        ("build a recommendation system", "recommendation"),
        ("recommend products to users", "recommendation"),
        ("movie recommendation engine", "recommendation"),
        ("collaborative filtering model", "recommendation"),
        ("product recommendation", "recommendation"),
        ("user-item matrix collaborative", "recommendation"),
        ("content-based filtering model", "recommendation"),
        ("suggest similar items to users", "recommendation"),
        ("build Netflix-like recommender", "recommendation"),
        ("playlist recommendation model", "recommendation"),
        ("item-based collaborative filtering", "recommendation"),
        ("matrix factorization SVD recommendation", "recommendation"),
        # nlp
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
        ("chatbot training data preparation", "nlp"),
        ("language model fine-tuning data", "nlp"),
        # general_ml
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
        ("end to end ML pipeline", "general_ml"),
        ("unsupervised learning clustering", "general_ml"),
    ]

    # Light augmentation
    augmented = list(TRAINING_DATA)
    synonyms = {"predict":"estimate","build":"create","detect":"identify",
                "model":"classifier","dataset":"data","prepare":"process",
                "forecast":"project"}
    for text, label in TRAINING_DATA:
        for w, syn in synonyms.items():
            if w in text.lower():
                augmented.append((text.lower().replace(w, syn), label))
                break

    texts  = [d[0] for d in augmented]
    labels = [d[1] for d in augmented]
    print(f"  Samples: {len(texts)} | Classes: {len(set(labels))}")

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=8000,
                                   sublinear_tf=True, strip_accents="unicode")),
        ("clf",   LogisticRegression(C=3.0, max_iter=2000,
                                      class_weight="balanced", solver="lbfgs"))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(pipeline, texts, y, cv=cv, scoring="accuracy")
    f1  = cross_val_score(pipeline, texts, y, cv=cv, scoring="f1_weighted")
    print(f"  CV Accuracy : {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"  CV F1       : {f1.mean():.3f} ± {f1.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(texts, y, test_size=0.2,
                                                random_state=42, stratify=y)
    pipeline.fit(X_tr, y_tr)
    print("\n  Test set report:")
    print(classification_report(y_te, pipeline.predict(X_te),
                                 target_names=le.classes_, zero_division=0))
    pipeline.fit(texts, y)   # retrain on full data

    save(pipeline, "intent_classifier.pkl")
    save(le,       "label_encoder.pkl")
    RESULTS["intent_classifier"] = {
        "cv_accuracy": round(float(acc.mean()), 4),
        "cv_f1":       round(float(f1.mean()),  4),
        "classes":     list(le.classes_),
        "n_samples":   len(texts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. FRAUD DETECTION  (on credit_card_fraud.csv)
# ─────────────────────────────────────────────────────────────────────────────
def train_fraud_detector():
    sep("2/6  Fraud Detector  [GradientBoosting + IsolationForest]")

    df = pd.read_csv(os.path.join(DATA_DIR, "credit_card_fraud.csv"))
    df = df.drop_duplicates()
    print(f"  Dataset: {df.shape} | Fraud rate: {df['Class'].mean():.2%}")

    feature_cols = [c for c in df.columns if c != 'Class']
    X = df[feature_cols].copy()
    y = df['Class'].values

    # Impute missing
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # SMOTE to handle class imbalance
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_tr_s, y_tr)
    print(f"  After SMOTE: {X_res.shape} | Fraud: {y_res.mean():.2%}")

    # GradientBoosting classifier
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8,
        min_samples_leaf=20, random_state=42
    )
    gb.fit(X_res, y_res)
    y_prob = gb.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob > 0.4).astype(int)   # lower threshold: recall > precision for fraud

    auc = roc_auc_score(y_te, y_prob)
    ap  = average_precision_score(y_te, y_prob)
    f1  = f1_score(y_te, y_pred, zero_division=0)
    print(f"\n  GradientBoosting Results:")
    print(f"    ROC-AUC : {auc:.4f}")
    print(f"    Avg Precision: {ap:.4f}")
    print(f"    F1-Score: {f1:.4f}")
    print(classification_report(y_te, y_pred,
                                 target_names=["Legit","Fraud"], zero_division=0))

    # IsolationForest (unsupervised layer)
    iso = IsolationForest(contamination=0.05, n_estimators=200,
                           random_state=42, n_jobs=-1)
    iso.fit(X_tr_s)

    # Save full pipeline
    fraud_pipeline = {"imputer": imp, "scaler": scaler,
                       "classifier": gb, "iso_forest": iso,
                       "feature_cols": feature_cols}
    save(fraud_pipeline, "fraud_detector.pkl")
    RESULTS["fraud_detector"] = {
        "roc_auc": round(auc, 4), "avg_precision": round(ap, 4),
        "f1": round(f1, 4), "dataset": "credit_card_fraud.csv",
        "n_train": int(X_res.shape[0]), "n_test": int(X_te.shape[0])
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHURN CLASSIFIER  (on customer_churn.csv)
# ─────────────────────────────────────────────────────────────────────────────
def train_churn_classifier():
    sep("3/6  Churn Classifier  [RandomForest + SMOTE]")

    df = pd.read_csv(os.path.join(DATA_DIR, "customer_churn.csv"))
    df = df.drop_duplicates()
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"  Dataset: {df.shape} | Churn rate: {(df['Churn']=='Yes').mean():.2%}")

    y = (df["Churn"] == "Yes").astype(int).values
    X = df.drop(columns=["Churn"])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_tr_pp = preprocessor.fit_transform(X_tr)
    X_te_pp = preprocessor.transform(X_te)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_tr_pp, y_tr)

    rf.fit(X_res, y_res)
    y_pred = rf.predict(X_te_pp)
    y_prob = rf.predict_proba(X_te_pp)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    f1  = f1_score(y_te, y_pred, zero_division=0)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(classification_report(y_te, y_pred,
                                 target_names=["No Churn","Churn"], zero_division=0))

    churn_pipeline = {"preprocessor": preprocessor, "model": rf,
                       "num_cols": num_cols, "cat_cols": cat_cols}
    save(churn_pipeline, "churn_classifier.pkl")
    RESULTS["churn_classifier"] = {
        "accuracy": round(acc, 4), "roc_auc": round(auc, 4),
        "f1": round(f1, 4), "dataset": "customer_churn.csv"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANOMALY DETECTOR  (trained on fraud data, generalizes)
# ─────────────────────────────────────────────────────────────────────────────
def train_anomaly_detector():
    sep("4/6  Anomaly Detector  [IsolationForest + LOF Ensemble]")

    df = pd.read_csv(os.path.join(DATA_DIR, "credit_card_fraud.csv"))
    df = df.drop_duplicates()
    y_true = df["Class"].values
    X = df.drop(columns=["Class"]).select_dtypes(include=[np.number])

    imp    = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp  = imp.fit_transform(X)
    X_sc   = scaler.fit_transform(X_imp)

    # Grid search contamination
    best_auc, best_cont, best_iso = 0, 0.05, None
    for cont in [0.03, 0.05, 0.07, 0.10]:
        iso = IsolationForest(contamination=cont, n_estimators=200,
                               random_state=42, n_jobs=-1)
        iso.fit(X_sc)
        scores = -iso.score_samples(X_sc)
        sc_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        auc = roc_auc_score(y_true, sc_norm)
        print(f"    contamination={cont:.2f} → AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_cont, best_iso = auc, cont, iso

    print(f"\n  Best IsolationForest: contamination={best_cont} AUC={best_auc:.4f}")

    # LOF evaluation
    lof = LocalOutlierFactor(contamination=best_cont, n_neighbors=20, n_jobs=-1)
    lof_preds = lof.fit_predict(X_sc)
    lof_scores = -lof.negative_outlier_factor_
    lof_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-9)
    lof_auc = roc_auc_score(y_true, lof_norm)
    lof_f1  = f1_score(y_true, (lof_preds == -1).astype(int), zero_division=0)
    print(f"  LOF: AUC={lof_auc:.4f} | F1={lof_f1:.4f}")

    # Ensemble score (weighted)
    iso_scores_norm = -best_iso.score_samples(X_sc)
    iso_norm = (iso_scores_norm - iso_scores_norm.min()) / (iso_scores_norm.max() - iso_scores_norm.min() + 1e-9)
    ensemble = 0.6 * iso_norm + 0.4 * lof_norm
    ens_auc = roc_auc_score(y_true, ensemble)
    print(f"  Ensemble (0.6 IF + 0.4 LOF): AUC={ens_auc:.4f}")

    anomaly_pkg = {
        "imputer":       imp,
        "scaler":        scaler,
        "iso_forest":    best_iso,
        "best_cont":     best_cont,
        "iso_auc":       round(best_auc, 4),
        "lof_auc":       round(lof_auc,  4),
        "ensemble_auc":  round(ens_auc,  4),
    }
    save(anomaly_pkg, "anomaly_detector.pkl")
    save(scaler, "anomaly_scaler.pkl")
    save(best_iso, "isolation_forest.pkl")
    RESULTS["anomaly_detector"] = {
        "iso_auc": round(best_auc, 4), "lof_auc": round(lof_auc, 4),
        "ensemble_auc": round(ens_auc, 4), "best_contamination": best_cont,
        "dataset": "credit_card_fraud.csv"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. HOUSE PRICE REGRESSOR  (on house_prices.csv)
# ─────────────────────────────────────────────────────────────────────────────
def train_house_price_regressor():
    sep("5/6  House Price Regressor  [GradientBoosting]")

    df = pd.read_csv(os.path.join(DATA_DIR, "house_prices.csv"))
    df = df.drop_duplicates().drop(columns=["Id"], errors="ignore")
    print(f"  Dataset: {df.shape}")

    y = df["SalePrice"].values
    X = df.drop(columns=["SalePrice"])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_tr_pp = preprocessor.fit_transform(X_tr)
    X_te_pp = preprocessor.transform(X_te)

    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8,
        min_samples_leaf=10, random_state=42
    )
    gb.fit(X_tr_pp, np.log1p(y_tr))   # log transform target (skewed prices)
    y_pred_log = gb.predict(X_te_pp)
    y_pred = np.expm1(y_pred_log)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")

    reg_pipeline = {"preprocessor": preprocessor, "model": gb,
                     "num_cols": num_cols, "cat_cols": cat_cols,
                     "log_transform": True}
    save(reg_pipeline, "house_price_regressor.pkl")
    RESULTS["house_price_regressor"] = {
        "mae": round(mae, 2), "rmse": round(rmse, 2),
        "r2": round(r2, 4), "dataset": "house_prices.csv"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. CANCER CLASSIFIER  (on breast_cancer.csv)
# ─────────────────────────────────────────────────────────────────────────────
def train_cancer_classifier():
    sep("6/6  Cancer Classifier  [RandomForest + SVM Ensemble]")

    df = pd.read_csv(os.path.join(DATA_DIR, "breast_cancer.csv"))
    df = df.drop_duplicates()
    print(f"  Dataset: {df.shape}")

    le_target = LabelEncoder()
    y = le_target.fit_transform(df["diagnosis"].values)   # M=1, B=0
    X = df.drop(columns=["diagnosis"]).select_dtypes(include=[np.number])

    imp    = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp  = imp.fit_transform(X)
    X_sc   = scaler.fit_transform(X_imp)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]
    rf_pred = rf.predict(X_te)
    rf_auc  = roc_auc_score(y_te, rf_prob)
    rf_f1   = f1_score(y_te, rf_pred, zero_division=0)
    print(f"  RandomForest → AUC={rf_auc:.4f} | F1={rf_f1:.4f}")

    # SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale",
               class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_tr, y_tr)
    svm_prob = svm.predict_proba(X_te)[:, 1]
    svm_pred = svm.predict(X_te)
    svm_auc  = roc_auc_score(y_te, svm_prob)
    svm_f1   = f1_score(y_te, svm_pred, zero_division=0)
    print(f"  SVM          → AUC={svm_auc:.4f} | F1={svm_f1:.4f}")

    # Soft ensemble
    ens_prob = 0.5 * rf_prob + 0.5 * svm_prob
    ens_pred = (ens_prob > 0.5).astype(int)
    ens_auc  = roc_auc_score(y_te, ens_prob)
    ens_f1   = f1_score(y_te, ens_pred, zero_division=0)
    print(f"  Ensemble     → AUC={ens_auc:.4f} | F1={ens_f1:.4f}")
    print(classification_report(y_te, ens_pred,
                                 target_names=le_target.classes_, zero_division=0))

    cancer_pipeline = {
        "imputer": imp, "scaler": scaler,
        "rf": rf, "svm": svm,
        "le": le_target, "feature_cols": list(X.columns)
    }
    save(cancer_pipeline, "cancer_classifier.pkl")
    RESULTS["cancer_classifier"] = {
        "rf_auc": round(rf_auc, 4), "svm_auc": round(svm_auc, 4),
        "ensemble_auc": round(ens_auc, 4), "ensemble_f1": round(ens_f1, 4),
        "dataset": "breast_cancer.csv"
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  DataForge AI — Full ML Training Pipeline")
    print("  Training 6 models on 5 real datasets\n")

    train_intent_classifier()
    train_fraud_detector()
    train_churn_classifier()
    train_anomaly_detector()
    train_house_price_regressor()
    train_cancer_classifier()

    # Save combined results report
    report_path = os.path.join(SAVE_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(RESULTS, f, indent=2)

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE — Summary")
    print(f"{'='*60}")
    for model, metrics in RESULTS.items():
        print(f"\n  {model}:")
        for k, v in metrics.items():
            if k not in ("classes", "feature_cols", "cat_cols", "num_cols"):
                print(f"    {k}: {v}")
    print(f"\n  📄 Full report: ml/saved_models/training_report.json")
    print(f"  📦 All models:  ml/saved_models/  +  backend/models/")
    print(f"{'='*60}\n")
