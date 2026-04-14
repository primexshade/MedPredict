# Changelog

All notable changes to **MedPredict** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `scripts/augment_data.py` — per-class multivariate Gaussian synthetic data augmentation for all 4 disease datasets
- MLflow model versions 2 for all 4 diseases after training on augmented data

### Changed
- Expanded dataset sizes: Heart 1,025→1,999, Diabetes 768→1,499, Cancer 569→1,100, Kidney 400→849 rows
- Retrained all 4 models with augmented data; improved Heart AUC-PR from baseline to **0.9439**

---

## [1.0.0] — 2026-02-01

### Added

#### Machine Learning
- Full training pipeline with **Optuna** Bayesian hyperparameter search (50–100 trials)
- **Nested cross-validation** (5-fold outer × 3-fold inner) for unbiased performance estimation
- **CalibratedClassifierCV** with isotonic regression for probability calibration
- MLflow experiment tracking and model registry (local file-store)
- 4 disease models trained and registered:
  - `disease-prediction-heart` (XGBoost, AUC-PR: 0.9439)
  - `disease-prediction-diabetes` (LightGBM, AUC-PR: 0.7790)
  - `disease-prediction-cancer` (XGBoost, AUC-PR: 0.9931)
  - `disease-prediction-kidney` (XGBoost, AUC-PR: 1.0000)
- Domain-aware feature engineering for all 4 diseases (ACC/AHA, ADA, KDIGO guidelines)
- SMOTE / BorderlineSMOTE oversampling for class-imbalanced datasets

#### API
- FastAPI 0.111 backend with async Uvicorn ASGI server
- JWT authentication (HS256, access=30min, refresh=7d)
- Role-based access control (admin / clinician / viewer)
- Prediction endpoints: `POST /api/v1/predict/{heart,diabetes,cancer,kidney}`
- Analytics stub endpoints: `GET /api/v1/analytics/{summary,clusters,comorbidity-rules}`
- Patient registry endpoints: `GET /api/v1/patients/`
- Redis response caching (5-minute TTL, SHA-256 keying, graceful no-op fallback)
- SlowAPI rate limiting (20 req/min on prediction endpoints)
- PredictionEngine singleton with SHAP TreeExplainer for every prediction
- Model preloading at startup (zero cold-start latency)

#### Frontend
- React 18 + TypeScript + Vite application
- 5 pages: Login, Dashboard, Predict, Analytics, Patients
- Recharts area chart (prediction trends), radar chart (risk overview), scatter chart (clusters)
- Disease-specific prediction forms with real-time range validation
- SHAP feature contribution display with direction arrows
- Risk category badges (LOW → CRITICAL, color coded)
- Clinical action recommendations panel
- Axios interceptors for auto-injecting Bearer tokens and auto-redirecting on 401

#### Infrastructure
- Multi-stage Dockerfile (builder + slim production image)
- Docker Compose stack: api, frontend (nginx), postgres, redis, mlflow, nginx reverse proxy
- GitHub Actions CI: ruff lint, mypy type check, pytest unit + integration, docker build
- Alembic migration setup with 4 tables: users, patients, predictions, audit_logs
- Pydantic Settings with .env file support

#### Documentation
- `docs/PROJECT_DOCUMENTATION.md` — 942-line complete technical reference
- `docs/API_REFERENCE.md` — full REST API specification with examples
- `docs/CONTRIBUTING.md` — contributor guide with workflow, standards, and new-model tutorial
- `docs/DEPLOYMENT.md` — Docker, production, and cloud deployment guide
- `docs/SECURITY.md` — security policies and vulnerability reporting
- `docs/RUNBOOK.md` — operational runbook for on-call engineers
- `docs/ARCHITECTURE.md` — system architecture deep-dive
- `docs/model_cards/` — per-disease ML model cards
- `README.md` — project overview and quick-start

#### Data
- Download scripts for all 4 public datasets (UCI, Kaggle)
- `scripts/run_training.py` CLI training entry point
- `scripts/augment_data.py` synthetic data augmentation

#### Testing
- 39 unit tests covering preprocessing, feature engineering, risk scoring, security
- Pytest configuration with asyncio mode

---

## [0.1.0] — 2025-10-01 (Internal Preview)

### Added
- Initial project scaffold
- Basic FastAPI application structure
- Preliminary sklearn preprocessing pipeline
- Prototype React dashboard (no auth)
