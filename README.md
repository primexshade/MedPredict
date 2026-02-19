# Disease Prediction & Health Risk Analysis Platform

> An intelligent healthcare analytics platform for multi-disease prediction, patient risk scoring, and clinical decision support — built with production-grade ML engineering practices.

## Overview

This platform applies advanced data mining and machine learning to predict multiple diseases (heart disease, diabetes, breast cancer, chronic kidney disease), segment patient populations, mine comorbidity patterns, and provide explainable AI outputs for clinical interpretability.

**Key Capabilities:**
- 🫀 Multi-disease risk prediction with calibrated probabilities
- 📊 Tiered risk scoring (LOW → CRITICAL) with confidence intervals
- 🧬 Patient clustering & phenotyping (GMM-based segmentation)
- 🔗 Comorbidity pattern discovery (FP-Growth association rules)
- 🔍 Explainable AI outputs (SHAP waterfall charts, plain-English summaries)
- 🖥️ React dashboard with real-time risk visualization
- 🔐 Role-based access control (Clinician, Admin, Researcher, Patient)

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost, LightGBM, Scikit-learn, SHAP |
| Experiment Tracking | MLflow |
| Hyperparameter Tuning | Optuna (Bayesian) |
| Backend API | FastAPI + Pydantic v2 |
| Database | PostgreSQL 16 + SQLAlchemy |
| Caching | Redis |
| Async Tasks | Celery |
| Frontend | React 18 + TypeScript + Recharts |
| Authentication | JWT + RBAC |
| Deployment | Docker + GitHub Actions + GCP Cloud Run |

## Project Structure

```
disease-prediction-platform/
├── src/
│   ├── data/          # Data loading, validation, splitting
│   ├── features/      # Preprocessing pipelines per disease
│   ├── models/        # Training, evaluation, calibration, registry
│   ├── mining/        # Clustering + association rule mining
│   ├── explainability/# SHAP + LIME explainers
│   ├── scoring/       # Risk scoring engine
│   ├── api/           # FastAPI routers, schemas
│   ├── auth/          # JWT + RBAC
│   ├── db/            # SQLAlchemy models + migrations
│   └── reports/       # PDF report generator
├── frontend/          # React dashboard
├── notebooks/         # EDA and exploration notebooks
├── tests/             # Unit, integration, e2e tests
├── docker/            # Dockerfiles + compose
└── scripts/           # Data download, DB seeding
```

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16
- Redis 7
- Node.js 20+ (for frontend)

### Backend Setup

```bash
# Clone and enter repo
git clone <your-repo-url>
cd disease-prediction-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your database URL, JWT secret, etc.

# Run database migrations
alembic upgrade head

# Start API server
uvicorn src.api.main:app --reload --port 8000
```

### Dataset Download

```bash
bash scripts/download_datasets.sh
```

### Model Training

```bash
# Train all disease models
python scripts/run_training.py --disease all

# Train a single disease model
python scripts/run_training.py --disease heart --trials 100
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev  # Starts on http://localhost:5173
```

### Docker (Full Stack)

```bash
docker-compose -f docker/docker-compose.yml up
```

## API Documentation

Once the server is running, interactive API docs are available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Model Performance

| Disease | AUC-ROC | AUC-PR | Sensitivity | Specificity |
|---|---|---|---|---|
| Heart Disease | 0.924 | 0.891 | 0.891 | 0.872 |
| Diabetes | 0.882 | 0.857 | 0.863 | 0.841 |
| Breast Cancer | 0.968 | 0.951 | 0.942 | 0.961 |
| Kidney Disease | 0.971 | 0.943 | 0.956 | 0.958 |

*Results on held-out test sets. See `docs/model_cards/` for detailed evaluation.*

## Git Workflow

- `main` — production-ready, protected (PR + CI required)
- `dev` — integration branch
- `feature/*` — individual features
- `fix/*` — bug fixes

Uses semantic commit messages: `feat:`, `fix:`, `refactor:`, `test:`, `chore:`, `docs:`

## Project Documentation

Full system design, architecture, and Git workflow strategy:
- [System Design Part 1](docs/system_design_part1.md) — Architecture, Data Strategy, ML Design
- [System Design Part 2](docs/system_design_part2.md) — Backend, Frontend, Roadmap, Git Workflow
- [API Reference](docs/api_reference.md)
- [Model Cards](docs/model_cards/)

## License

MIT License — for academic and research use.
