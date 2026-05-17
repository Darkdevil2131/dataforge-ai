# DataForge AI

AI-powered ML preprocessing platform. Upload any messy dataset, describe your goal — DataForge cleans it, engineers features, detects anomalies, and lets you download production-ready data.

---

## Quick Start

### Mac / Linux
```bash
chmod +x start.sh && ./start.sh
```

### Windows
```
Double-click  start.bat
```

Open **http://localhost:5173** in your browser. Done.

---

## Requirements

- Python 3.10+ → https://python.org
- Node.js 18+  → https://nodejs.org

---

## Manual Start

**Terminal 1 — Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
# Runs at http://localhost:8000  |  Docs: http://localhost:8000/docs
```

**Terminal 2 — Frontend**
```bash
cd frontend
npm install
npm run dev
# Runs at http://localhost:5173
```

---

## Trained Models (Real Data)

| Model | Algorithm | Dataset | Result |
|-------|-----------|---------|--------|
| Intent Classifier | TF-IDF + LogReg | 210 samples | CV 92.9% acc |
| Fraud Detector | GradientBoosting + SMOTE | 10K rows | AUC 1.00, F1 0.99 |
| Churn Classifier | RandomForest + SMOTE | 7K rows | Acc 71.7% |
| Anomaly Detector | IsolationForest + LOF | 10K rows | AUC 1.00 ensemble |
| House Price Regressor | GradientBoosting | 1.5K rows | R² 0.92 |
| Cancer Classifier | RF + SVM Ensemble | 569 rows | AUC 0.997, F1 0.976 |

Retrain anytime: `cd ml && python train_all_models.py`

---

## Project Structure

```
dataforge-ai/
├── start.sh / start.bat      ← Run to start everything
├── backend/
│   ├── main.py               ← FastAPI (7 endpoints)
│   ├── requirements.txt
│   ├── engines/              ← 6 specialist engines
│   ├── models/               ← Pre-trained .pkl files
│   └── utils/helpers.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── hooks/useDataForge.js
│   │   ├── components/       ← 4 components
│   │   └── utils/api.js
│   └── index.html
└── ml/
    ├── datasets/             ← 5 real datasets
    ├── saved_models/         ← Trained models + metrics
    └── train_all_models.py   ← Full training script
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/process` | POST | Full pipeline: clean + engineer + report |
| `/api/analyze` | POST | Quick schema analysis |
| `/api/recommend` | POST | Dataset recommendations |
| `/api/download/cleaned` | POST | Download cleaned CSV/Excel/Parquet |
| `/api/model-stats` | GET | Training metrics |
| `/health` | GET | Health check |

---

**Stack:** FastAPI · scikit-learn · imbalanced-learn · React 18 · Vite · Tailwind CSS
