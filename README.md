# MedPredict — AI-Powered Multi-Disease Risk Prediction Platform

> **A production-grade clinical decision support system** for automated risk stratification of heart disease, diabetes, breast cancer, and chronic kidney disease using calibrated probabilistic machine learning, explainable AI (SHAP), and a clinician-friendly React dashboard.

---

## Quick Start

```bash
# 1. Clone & create virtual environment
git clone <repo-url>
cd "Disease Prediction"
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Download datasets & train all 4 models
bash scripts/download_datasets.sh
MLFLOW_TRACKING_URI="file://$(pwd)/mlruns" python scripts/run_training.py --disease all

# 3. Start the API (models auto-load from mlruns/)
MLFLOW_TRACKING_URI="file://$(pwd)/mlruns" uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the frontend (in a second terminal)
cd frontend && npm install && npm run dev

# 5. Open in browser
#   Frontend: http://localhost:5173   (login: admin@example.com / admin)
#   API Docs: http://localhost:8000/docs
```

---

## What This Platform Does

MedPredict allows clinicians to:

1. **Enter patient biomarkers** (lab values, age, vitals) into a web form
2. **Run a trained ML model** that produces a calibrated disease probability
3. **Receive a composite risk score** (LOW → BORDERLINE → MODERATE → HIGH → CRITICAL)
4. **See SHAP explanations** — which specific factors drove the risk up or down
5. **Get a clinical action** — e.g., "Specialist referral, 1-month follow-up"
6. **Track population trends** — dashboards show aggregate prediction trends over time

---

## Supported Diseases

| Disease | Model | AUC-ROC | AUC-PR | Key Features |
|---|---|---|---|---|
| ❤️ Heart Disease | XGBoost (calibrated) | 0.862 | 0.889 | ECG, cholesterol, ST depression, angina |
| 🩸 Diabetes (Type 2) | LightGBM (calibrated) | 0.844 | 0.720 | Glucose, BMI, insulin, HbA1c proxy |
| 🎗️ Breast Cancer | XGBoost (calibrated) | 0.999 | 0.999 | WDBC cell morphology — 30 features |
| 🫘 Kidney Disease | XGBoost (calibrated) | 1.000 | 1.000 | eGFR proxy, creatinine, hemoglobin |

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite + TS)                   │
│  Login │ Dashboard │ Predict │ Analytics │ Patients              │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS / REST
┌────────────────────────▼────────────────────────────────────────┐
│               FastAPI Backend  (port 8000)                      │
│  /auth  /predict  /analytics  /patients  /reports               │
│  JWT auth │ Rate limiting │ CORS │ Redis caching                │
└────────────────────────┬────────────────────────────────────────┘
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌───▼──────┐
    │  MLflow   │  │ PostgreSQL │  │  Redis   │
    │  mlruns/  │  │  (Alembic) │  │  Cache   │
    └───────────┘  └───────────┘  └──────────┘
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, TypeScript, Vite, Recharts, CSS |
| **Backend API** | FastAPI, Pydantic v2, Uvicorn |
| **ML / Training** | XGBoost, LightGBM, scikit-learn, imbalanced-learn |
| **Hyperparameter Tuning** | Optuna (TPE sampler, nested CV) |
| **Explainability** | SHAP TreeExplainer |
| **Experiment Tracking** | MLflow (local file store) |
| **Preprocessing** | sklearn ColumnTransformer, MICE imputation, BorderlineSMOTE |
| **Database** | PostgreSQL + SQLAlchemy + Alembic migrations |
| **Caching** | Redis (async, with no-op fallback for local dev) |
| **Auth** | JWT (HS256) with RBAC (admin / clinician / viewer) |
| **Containerization** | Docker + docker-compose |
| **CI** | GitHub Actions |

---

## Project Structure

```
Disease Prediction/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app factory + lifespan startup
│   │   ├── deps.py              # JWT auth + Redis dependency injection
│   │   └── routers/
│   │       ├── predict.py       # Core prediction engine + 4 disease endpoints
│   │       ├── analytics.py     # Clustering + comorbidity association rules
│   │       ├── patients.py      # Patient CRUD + history
│   │       ├── auth.py          # Login / refresh-token endpoints
│   │       └── reports.py       # PDF/JSON diagnostic reports
│   ├── models/
│   │   └── train.py             # Optuna + nested CV + MLflow logging
│   ├── features/
│   │   ├── pipeline.py          # sklearn ColumnTransformer factory
│   │   └── engineering.py       # Domain-aware feature engineering (4 diseases)
│   ├── data/
│   │   └── load.py              # Dataset loaders (UCI, sklearn.datasets)
│   ├── scoring/
│   │   └── risk_scorer.py       # Composite risk scoring + confidence intervals
│   ├── explainability/
│   │   └── shap_explainer.py    # SHAP TreeExplainer wrapper
│   ├── auth/
│   │   └── security.py          # JWT create/decode + RBAC
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   └── session.py           # Async DB session factory
│   └── config.py                # Pydantic Settings (env-driven)
├── frontend/                    # React 18 + TypeScript SPA
├── scripts/
│   ├── download_datasets.sh     # Downloads all 4 raw datasets
│   └── run_training.py          # CLI training entry point
├── alembic/                     # Database migrations
├── docker/
│   ├── Dockerfile               # Multi-stage production image
│   └── docker-compose.yml       # Full stack: API + Postgres + Redis + Nginx
├── tests/                       # pytest unit + integration tests
├── mlruns/                      # MLflow experiment logs + model artifacts
└── docs/
    └── PROJECT_DOCUMENTATION.md # Comprehensive technical documentation
```

---

## API Endpoints Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | System health check |
| POST | `/api/v1/auth/login` | No | Issue JWT access + refresh tokens |
| POST | `/api/v1/auth/refresh` | Refresh token | Rotate access token |
| POST | `/api/v1/predict/heart` | JWT | Heart disease risk prediction |
| POST | `/api/v1/predict/diabetes` | JWT | Type 2 diabetes risk prediction |
| POST | `/api/v1/predict/cancer` | JWT | Breast cancer risk prediction |
| POST | `/api/v1/predict/kidney` | JWT | Chronic kidney disease prediction |
| GET | `/api/v1/analytics/summary` | JWT | Population-level risk summary |
| GET | `/api/v1/analytics/clusters` | JWT | Patient phenotype cluster data |
| GET | `/api/v1/analytics/comorbidities` | JWT | Association rules between diseases |
| GET/POST | `/api/v1/patients` | JWT | Patient registry CRUD |
| GET | `/api/v1/reports/{patient_id}` | JWT | Diagnostic report generation |

Full interactive docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## Running with Docker (Full Stack)

```bash
docker compose -f docker/docker-compose.yml up -d

# Services:
#   API:      http://localhost:8000
#   Frontend: http://localhost:3000
#   MLflow:   http://localhost:5001
#   Postgres: localhost:5432
#   Redis:    localhost:6379
```

---

## Model Training

```bash
# Train a single disease
python scripts/run_training.py --disease heart --trials 100

# Train all diseases sequentially
python scripts/run_training.py --disease all --trials 80

# View results in MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/mlruns
# Open: http://localhost:5001
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## Environment Variables

Copy `.env.example` → `.env` and set:

```env
MLFLOW_TRACKING_URI=file:///path/to/Disease Prediction/mlruns
JWT_SECRET=your-secret-key-min-32-chars
DATABASE_URL=postgresql://user:password@localhost:5432/disease_prediction
REDIS_URL=redis://localhost:6379/0
ENVIRONMENT=development
```

---

## Security

- All prediction endpoints require a valid **JWT Bearer** token
- **RBAC**: `admin` (full access), `clinician` (predict + read), `viewer` (read-only)
- Rate limiting: 100 req/min general, 20 req/min prediction endpoints
- Passwords hashed with **bcrypt**
- Tokens signed with **HS256** (configurable to RS256 for production)

---

## Documentation

See [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) for the complete technical deep-dive covering every module, ML decision, feature engineering rationale, API schema, and deployment guide.
