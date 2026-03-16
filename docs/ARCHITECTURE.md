# System Architecture — MedPredict

This document provides a deep-dive into the technical architecture of MedPredict, covering system design decisions, data flow, component interactions, and the rationale behind key choices.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Backend Architecture](#2-backend-architecture)
3. [Machine Learning Architecture](#3-machine-learning-architecture)
4. [Frontend Architecture](#4-frontend-architecture)
5. [Data Architecture](#5-data-architecture)
6. [Security Architecture](#6-security-architecture)
7. [Request Lifecycle](#7-request-lifecycle)
8. [Design Decisions](#8-design-decisions)

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  Browser: React 18 + TypeScript + Recharts (Vite dev / Nginx production)    │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ HTTPS / REST JSON
                                     │ Bearer JWT in Authorization header
┌────────────────────────────────────▼─────────────────────────────────────────┐
│                         NGINX REVERSE PROXY                                  │
│  TLS termination · Rate limiting pass-through · Static file serving         │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ HTTP → localhost:8000
┌────────────────────────────────────▼─────────────────────────────────────────┐
│                     FASTAPI APPLICATION (Uvicorn ASGI)                      │
│                                                                              │
│   ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────┐  │
│   │  CORSMiddle │   │  SlowAPI Rate│   │  JWT Middleware│   │  Lifespan  │  │
│   │  ware       │   │  Limiter     │   │  (Dep. Inject) │   │  (Startup) │  │
│   └─────────────┘   └──────────────┘   └───────────────┘   └────────────┘  │
│                                                                              │
│   ┌────────────┐  ┌───────────────┐  ┌────────────┐  ┌──────────────────┐  │
│   │ /auth      │  │ /predict      │  │ /analytics │  │ /patients        │  │
│   │ Router     │  │ Router        │  │ Router     │  │ Router           │  │
│   └────────────┘  └───────┬───────┘  └────────────┘  └──────────────────┘  │
│                           │                                                  │
│              ┌────────────▼────────────┐                                    │
│              │    PredictionEngine     │                                    │
│              │  ┌───────────────────┐  │                                    │
│              │  │ Model Registry    │  │  ← MLflow file store / server      │
│              │  │ heart → CalibrCV  │  │                                    │
│              │  │ diabetes → CalibrCV│ │                                    │
│              │  │ cancer → CalibrCV │  │                                    │
│              │  │ kidney → CalibrCV │  │                                    │
│              │  └───────────────────┘  │                                    │
│              │  ┌───────────────────┐  │                                    │
│              │  │ SHAP Explainers   │  │                                    │
│              │  │ per disease       │  │                                    │
│              │  └───────────────────┘  │                                    │
│              │  ThreadPoolExecutor(4)  │  ← CPU work offloaded from async   │
│              └─────────────────────────┘                                    │
│                                                                              │
│   ┌──────────────────────┐   ┌────────────────────────────────────────────┐ │
│   │  RiskScorer          │   │  DependencyInjection (deps.py)             │ │
│   │  Composite score     │   │  get_current_user() → JWT decode           │ │
│   │  Wilson CI           │   │  get_redis() → aioredis / NoOpRedis stub   │ │
│   │  Category mapping    │   │  require_permission() → RBAC check         │ │
│   └──────────────────────┘   └────────────────────────────────────────────┘ │
└──────────────────────────────────┬───────────────────────────────────────────┘
              ┌────────────────────┼──────────────────────────┐
              │                    │                          │
    ┌─────────▼──────┐   ┌─────────▼──────┐      ┌──────────▼─────┐
    │  PostgreSQL 15  │   │  Redis 7        │      │ MLflow Store   │
    │  + Alembic      │   │  (async, TTL=5m)│      │ (local / GCS)  │
    │  users          │   │  predict:*:sha  │      │ models/        │
    │  patients        │   │  cache keys    │      │ experiments/   │
    │  predictions     │   └────────────────┘      └────────────────┘
    │  audit_logs      │
    └─────────────────┘
```

---

## 2. Backend Architecture

### Application Factory Pattern

`src/api/main.py` uses an application factory (`create_app()`) that:
1. Creates a FastAPI instance with metadata
2. Registers middleware (CORS, rate limiting)
3. Mounts routers under `/api/v1`
4. Attaches a lifespan context manager (`@asynccontextmanager`) that:
   - On startup: calls `prediction_engine.preload_models()` (async)
   - On shutdown: logs graceful exit

This pattern enables clean testing (create fresh app per test) and clear separation of concerns.

### Dependency Injection (`src/api/deps.py`)

FastAPI's dependency injection system provides three shared dependencies:

```python
# JWT validation — used on every protected endpoint
async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    return {"user_id": payload["sub"], "role": payload["role"]}

# Redis — with graceful fallback
async def get_redis() -> AsyncGenerator:
    try:
        redis = aioredis.from_url(settings.redis_url)
        yield redis
    except Exception:
        yield _NoOpRedis()  # In-memory no-op when Redis unavailable

# RBAC factory
def require_permission(permission: str):
    async def check(current_user=Depends(get_current_user)):
        if not _has_permission(current_user["role"], permission):
            raise HTTPException(403)
    return check
```

### Async Architecture

The API is fully async (FastAPI + Uvicorn). However, ML inference (XGBoost/LightGBM + SHAP) is **CPU-bound and not async-safe**. This is handled by offloading to a `ThreadPoolExecutor`:

```python
_executor = ThreadPoolExecutor(max_workers=4)

async def predict(self, disease: str, df: DataFrame) -> tuple:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        partial(self._predict_sync, disease, df)
    )
```

This prevents the async event loop from being blocked by long-running matrix operations.

### Caching Strategy

```
Request → SHA256(sorted features)[:16] → "predict:heart:abc123456789abcd"
              │
      Redis GET ──→ HIT: deserialize and return (< 1ms)
              │
          MISS: run inference → Redis SETEX(key, 300, result)
```

Cache invalidation is time-based (5 minutes). Clinical correctness is maintained because the cache key excludes `patient_id` — only feature values determine the cache key. Two patients with identical biomarkers will share a cached result (the model output is deterministic given features).

---

## 3. Machine Learning Architecture

### Training Pipeline

```
Raw CSV
  │
  ├── load_<disease>()          # Schema validation, dtype coercion, constraint checks
  │
  ├── apply_feature_engineering()  # Clinical feature derivation (per-disease)
  │
  ├── Stratified Train/Val/Test split (70% / 15% / 15%)
  │
  ├── Outer Nested CV (5-fold)
  │     └── Inner Optuna (n_trials/5 per fold)
  │           └── build_full_pipeline(disease, model, X_train)
  │                 ├── ColumnTransformer
  │                 │     ├── Numeric: IterativeImputer → RobustScaler
  │                 │     └── Categorical: SimpleImputer → OneHotEncoder
  │                 ├── [SMOTE / BorderlineSMOTE]  ← training only
  │                 └── XGBClassifier / LGBMClassifier
  │
  ├── Final Optuna (full n_trials on train set)
  │
  ├── CalibratedClassifierCV(cv=5, method="isotonic")  # Probability calibration
  │
  ├── compute_metrics(y_test, y_proba, y_pred)  # AUC-PR, AUC-ROC, Sensitivity, etc.
  │
  └── MLflow: log_params, log_metrics, log_model, register_model
```

### Feature Pipeline

The `ImbPipeline` (from `imbalanced-learn`) is used rather than sklearn's `Pipeline` because:

- **sklearn Pipeline** applies all steps during `fit()` AND `transform()`. If SMOTE was a step, it would generate synthetic samples during cross-validation's validation transforms, causing **data leakage**.
- **ImbPipeline** correctly applies sampling steps only during `fit()`, never during `transform()` or `predict()`.

### Calibration Architecture

```
Raw XGBoost/LightGBM output → .predict_proba() → [0.0, 1.0]
          (NOT a probability — it's an uncalibrated score)
                    │
CalibratedClassifierCV(cv=5, method="isotonic")
          │
          ├── Splits training data into 5 folds
          ├── Fits 5 {pipeline + isotonic calibrator} pairs
          └── At inference: averages predictions across 5 calibrators
                    │
                    └── Calibrated probability (P(disease | features))
                        — genuinely interpretable as disease prevalence
```

### SHAP Architecture

```
CalibratedClassifierCV (loaded from MLflow)
    │
    ├── .calibrated_classifiers_[0]
    │         └── .estimator  ← ImbPipeline
    │                   └── .named_steps["classifier"]  ← XGBClassifier
    │                                                    ← used for SHAP
    │
    └── .calibrated_classifiers_[0].estimator.named_steps["preprocessor"]
                                                    ← used to transform background data
```

SHAP TreeExplainer needs:
1. The **inner tree model** (not the full pipeline or calibrator)
2. A **background dataset** pre-transformed through the preprocessor

This double extraction is necessary because SHAP operates on the model's feature space (after preprocessing), not on the raw clinical feature space.

---

## 4. Frontend Architecture

```
frontend/src/
├── main.tsx                  # App entry point (React DOM render)
├── App.tsx                   # Router setup (react-router-dom v6)
├── pages/
│   ├── Login.tsx             # Auth form → POST /auth/login
│   ├── Dashboard.tsx         # Population overview (charts)
│   ├── Predict.tsx           # Per-patient prediction (core clinical tool)
│   ├── Analytics.tsx         # Cluster scatter, comorbidity rules
│   └── Patients.tsx          # Patient registry table
├── services/
│   └── api.ts                # Axios instance, interceptors, all API types
└── components/               # Shared UI components
```

### State Management

The application uses **React local state + React Router** without a global state manager (no Redux/Zustand). This is appropriate because:

- Auth state: stored in `localStorage` (access_token, refresh_token) + React state
- Prediction results: local state within `Predict.tsx`
- List data (patients, analytics): fetched on mount, stored in local state

Global state is not needed because pages are largely independent — the Dashboard, Predict, Analytics, and Patients pages don't share mutable state.

### Auth Flow

```
Login.tsx → POST /auth/login
         → receives { access_token, refresh_token }
         → localStorage.setItem("access_token", ...)
         → navigate("/dashboard")

api.ts interceptors:
  Request: inject "Authorization: Bearer <token>" on every request
  Response (401): clear tokens → window.location.href = "/login"
```

---

## 5. Data Architecture

### Database Schema

```sql
users (id UUID, email, hashed_password, role, is_active, created_at)
   │
   ├── predictions.created_by FK
   └── patients.created_by FK

patients (id UUID, external_id[hospital MRN], full_name, date_of_birth, sex)
   │
   └── predictions.patient_id FK (ON DELETE CASCADE)

predictions (
   id UUID, patient_id FK, disease VARCHAR,
   input_features JSONB,      ← raw biomarker values as submitted
   calibrated_prob FLOAT,
   composite_score FLOAT,
   risk_category VARCHAR,
   shap_values JSONB,          ← top-5 feature contributions
   model_version VARCHAR,
   created_by FK, created_at
)

audit_logs (
   id BIGSERIAL,               ← append-only, never deleted
   user_id FK, action, resource, detail JSONB, ip_address INET, created_at
)
```

### MLflow Data Model

```
mlruns/
├── models/
│   ├── disease-prediction-heart/    ← RegisteredModel
│   │   ├── version-1/               ← ModelVersion
│   │   └── version-2/               ← after augmented retraining
│   ├── disease-prediction-diabetes/
│   ├── disease-prediction-cancer/
│   └── disease-prediction-kidney/
└── <experiment-id>/
    └── <run-id>/
        ├── params/        ← Optuna best hyperparams
        ├── metrics/       ← AUC-PR, AUC-ROC, Sensitivity, etc.
        └── artifacts/
            └── model/     ← sklearn CalibratedClassifierCV (pickle)
```

---

## 6. Security Architecture

```
Client → HTTPS (TLS 1.3) → Nginx
                              │
                     JWT Validation (deps.py get_current_user)
                     RBAC check (deps.py require_permission)
                     Rate limit (SlowAPI per IP)
                     Input validation (Pydantic v2 strict types)
                              │
                     Parametrized SQLAlchemy queries (no SQL injection)
                     Bcrypt password hashing (never plaintext)
                     Audit log (append-only, every sensitive action recorded)
```

---

## 7. Request Lifecycle

### Full Prediction Request (Cache Miss)

```
1. Client → POST /api/v1/predict/heart  (JWT in header)
2. CORS middleware → pass
3. SlowAPI → check rate limit → pass (< 20/min)
4. FastAPI router → HeartDiseaseInput.model_validate(body) ← Pydantic strict validation
5. Depends(get_current_user) → decode JWT → return {user_id, role, jti}
6. Depends(get_redis) → connect aioredis (or NoOpRedis)
7. _run_prediction("heart", payload.model_dump(), redis):
   a. cache_key = SHA256(sorted features)[:16]
   b. Redis GET → MISS
   c. df = pd.DataFrame([data])
   d. await prediction_engine.predict("heart", df)  ← async
      └── loop.run_in_executor(_executor, _predict_sync):
          i. apply_feature_engineering("heart", df)  → +7 clinical features
          ii. model.predict_proba(df_engineered)[:, 1]  → calibrated_prob
          iii. preprocessor.transform(df_engineered)  → X_transformed
          iv. shap.TreeExplainer.shap_values(X_df)  → shap_row
          v. rank top-5 by |shap_value|
   e. RiskScorer("heart").compute(prob)
      └── composite = 0.70 × prob (first visit, no velocity)
      └── Wilson CI
      └── categorize(composite) → RiskCategory.HIGH
      └── CLINICAL_ACTIONS[HIGH]
   f. Build PredictionResponse
   g. Redis SETEX(key, 300, response_json)
8. Return JSON response  ← ~200ms (no cache)
```

---

## 8. Design Decisions

| Decision | Chosen approach | Alternatives considered | Rationale |
|---|---|---|---|
| ML framework | XGBoost + LightGBM | Neural networks (LSTM, MLP) | Tree models are interpretable via SHAP, faster to train, don't require large datasets |
| HP tuning | Optuna (Bayesian) | GridSearchCV, RandomSearch | Bayesian search finds better hyperparams in fewer trials; Optuna's pruning kills bad trials early |
| Calibration | Isotonic regression (cv=5) | Platt scaling | Isotonic is non-parametric; better for non-S-shaped calibration curves common in clinical data |
| Explainability | SHAP TreeExplainer | LIME, attention weights | SHAP has theoretical guarantees (Shapley values); TreeExplainer is exact and fast for tree models |
| Imbalance handling | SMOTE / BorderlineSMOTE | Class weights, threshold tuning | SMOTE creates new data points; combined with `class_weight` in XGBoost via `scale_pos_weight` |
| Imputation | IterativeImputer (MICE) | Mean/median imputation | MICE preserves inter-feature correlations; better predictions when missingness is informative |
| Scaling | RobustScaler | StandardScaler, MinMaxScaler | Clinical data has legitimate outliers; RobustScaler uses IQR and is not distorted by them |
| Async inference | ThreadPoolExecutor | ProcessPoolExecutor, Celery | Thread pool is simpler than processes (shared memory for model); Celery adds operational complexity |
| Caching | Redis (content-addressed) | Memcached, in-memory LRU | Redis persists across API restarts; content addressing means cache is patient-independent |
| Auth | JWT (HS256) | Session cookies, OAuth2 + external IdP | JWT is stateless (no DB lookup per request); appropriate for a self-contained clinical tool |
