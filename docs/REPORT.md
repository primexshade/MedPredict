# MedPredict — Comprehensive Project Report

## Executive Summary

MedPredict is a **production-grade clinical decision support system** that provides AI-powered multi-disease risk prediction for Heart Disease, Type 2 Diabetes, Breast Cancer, and Chronic Kidney Disease. The platform combines calibrated probabilistic machine learning models with SHAP explainability, delivering actionable risk assessments through a modern React dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Machine Learning Pipeline](#4-machine-learning-pipeline)
5. [Backend API](#5-backend-api)
6. [Frontend Dashboard](#6-frontend-dashboard)
7. [Database Design](#7-database-design)
8. [Authentication & Security](#8-authentication--security)
9. [Data Mining Features](#9-data-mining-features)
10. [Deployment](#10-deployment)
11. [Testing Strategy](#11-testing-strategy)
12. [Bug Fixes Applied](#12-bug-fixes-applied)
13. [Future Enhancements](#13-future-enhancements)

---

## 1. Project Overview

### 1.1 Problem Statement

Clinical decision-making often relies on physician intuition and standard guidelines, which may not account for complex multi-variable interactions. MedPredict addresses this by:

- Providing **quantified risk probabilities** with confidence intervals
- Offering **explainable AI outputs** (SHAP values) so clinicians understand *why* a prediction was made
- Enabling **population-level analytics** through clustering and association rule mining
- Delivering a **clinician-friendly interface** that integrates into existing workflows

### 1.2 Supported Diseases

| Disease | Model | AUC-ROC | AUC-PR | Dataset |
|---------|-------|---------|--------|---------|
| ❤️ Heart Disease | XGBoost (calibrated) | 0.862 | 0.889 | UCI Cleveland (1,025 samples) |
| 🩸 Type 2 Diabetes | LightGBM (calibrated) | 0.844 | 0.720 | Pima Indians (768 samples) |
| 🎗️ Breast Cancer | XGBoost (calibrated) | 0.999 | 0.999 | WDBC (569 samples) |
| 🫘 Chronic Kidney Disease | XGBoost (calibrated) | 1.000 | 1.000 | UCI CKD (400 samples) |

### 1.3 Risk Categories

The platform maps calibrated probabilities to clinical risk tiers:

| Category | Probability Range | Clinical Action |
|----------|-------------------|-----------------|
| LOW | 0% – 20% | Routine follow-up in 12 months |
| BORDERLINE | 21% – 40% | Lifestyle counseling, 6-month recheck |
| MODERATE | 41% – 60% | Diagnostic workup, 3-month follow-up |
| HIGH | 61% – 80% | Specialist referral, 1-month follow-up |
| CRITICAL | 81% – 100% | Urgent intervention, immediate follow-up |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite + TS)                   │
│  Login │ Dashboard │ Predict │ Analytics │ Patients            │
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

### 2.1 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Frontend** | User interface, form handling, data visualization |
| **FastAPI** | REST API, authentication, request validation, ML inference orchestration |
| **MLflow** | Model versioning, experiment tracking, artifact storage |
| **PostgreSQL** | Persistent storage (users, patients, predictions, audit logs) |
| **Redis** | Session caching, prediction result caching, token blacklist |

---

## 3. Technology Stack

### 3.1 Frontend

| Technology | Purpose |
|------------|---------|
| **React 18.3** | UI component library with concurrent rendering |
| **TypeScript 5.5** | Static typing for maintainability |
| **Vite 5.4** | Fast HMR development server, optimized builds |
| **TanStack React Query** | Server state management, caching, automatic refetching |
| **Recharts** | Data visualization (area charts, radar, bar, pie, scatter) |
| **React Router v6** | Client-side routing with protected routes |
| **Axios** | HTTP client with interceptors for auth |

### 3.2 Backend

| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance async web framework |
| **Pydantic v2** | Data validation, serialization, settings management |
| **Uvicorn** | ASGI server with HTTP/2 support |
| **SQLAlchemy 2.0** | Async ORM with type annotations |
| **Alembic** | Database migration management |
| **structlog** | Structured JSON logging |
| **slowapi** | Rate limiting middleware |

### 3.3 Machine Learning

| Technology | Purpose |
|------------|---------|
| **XGBoost 2.0** | Gradient boosting for tabular data |
| **LightGBM 4.1** | Fast gradient boosting (diabetes model) |
| **scikit-learn 1.4** | Preprocessing, calibration, evaluation |
| **imbalanced-learn** | SMOTE oversampling for class imbalance |
| **Optuna 3.4** | Bayesian hyperparameter optimization |
| **SHAP 0.44** | TreeExplainer for feature importance |
| **MLflow 2.10** | Experiment tracking, model registry |

### 3.4 Infrastructure

| Technology | Purpose |
|------------|---------|
| **PostgreSQL** | Primary database (ACID, JSONB support) |
| **Redis** | Caching layer, session store, token blacklist |
| **Docker** | Containerization |
| **GitHub Actions** | CI/CD pipeline |

---

## 4. Machine Learning Pipeline

### 4.1 Data Preprocessing

```python
# Pipeline structure (src/features/pipeline.py)
ColumnTransformer([
    ('numeric', Pipeline([
        ('imputer', IterativeImputer()),  # MICE imputation
        ('scaler', StandardScaler()),
    ]), numeric_features),
    ('categorical', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder()),
    ]), categorical_features),
])
```

**Key preprocessing steps:**
1. **MICE Imputation** — Multivariate Imputation by Chained Equations for missing values
2. **Standard Scaling** — Z-score normalization for numeric features
3. **Ordinal Encoding** — Integer encoding for categorical variables
4. **Feature Engineering** — Disease-specific derived features (e.g., age-cholesterol ratio)

### 4.2 Feature Engineering

Each disease has domain-specific engineered features:

**Heart Disease:**
- `age_chol_ratio` = cholesterol / age
- `bp_hr_interaction` = resting BP × max heart rate
- `st_exercise_score` = oldpeak × slope

**Diabetes:**
- `bmi_glucose_interaction` = BMI × glucose
- `insulin_resistance_proxy` = glucose × insulin / 405
- `metabolic_risk_score` = composite of BMI, glucose, BP

**Cancer (WDBC):**
- `worst_to_mean_ratio_radius` = radius_worst / radius_mean
- `symmetry_compactness_product` = symmetry × compactness

**Kidney:**
- `egfr_proxy` = estimated GFR from creatinine and age
- `anemia_score` = hemoglobin / normal_range

### 4.3 Class Imbalance Handling

```python
# BorderlineSMOTE for minority class oversampling
from imblearn.over_sampling import BorderlineSMOTE
smote = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 4.4 Hyperparameter Optimization

```python
# Optuna TPE sampler with nested CV
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=10),
)
study.optimize(objective, n_trials=100)
```

**Search space example (XGBoost):**
- `max_depth`: [3, 12]
- `learning_rate`: [0.01, 0.3]
- `n_estimators`: [100, 1000]
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]

### 4.5 Probability Calibration

```python
# Platt scaling for probability calibration
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(
    base_estimator=xgb_model,
    method='sigmoid',  # Platt scaling
    cv=5,
)
```

### 4.6 SHAP Explainability

```python
# TreeExplainer for gradient boosting models
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Generate natural language summary
contributions = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
summary = generate_plain_english_summary(contributions[:5])
```

**Output format:**
```json
{
  "top_features": [
    {"feature": "cp", "shap_value": 0.312, "direction": "increases_risk"},
    {"feature": "thalach", "shap_value": -0.198, "direction": "decreases_risk"}
  ],
  "plain_english_summary": "The patient's atypical chest pain (cp=2) strongly increases risk. However, their high maximum heart rate (154 bpm) is protective."
}
```

---

## 5. Backend API

### 5.1 API Structure

```
/api/v1/
├── auth/
│   ├── POST /login       # JWT token pair issuance
│   ├── POST /refresh     # Token refresh
│   └── POST /logout      # Token blacklist
├── predict/
│   ├── POST /heart       # Heart disease prediction
│   ├── POST /diabetes    # Diabetes prediction
│   ├── POST /cancer      # Breast cancer prediction
│   └── POST /kidney      # Kidney disease prediction
├── analytics/
│   ├── GET /summary      # Population statistics
│   ├── GET /clusters     # Patient phenotype clusters
│   └── GET /comorbidity-rules  # Association rules
├── patients/
│   ├── GET /             # List patients (paginated)
│   ├── GET /{id}         # Get patient by ID
│   └── POST /            # Create patient
└── reports/
    └── GET /{patient_id} # Generate patient report
```

### 5.2 Request/Response Examples

**Prediction Request:**
```json
POST /api/v1/predict/heart
{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 140,
  "chol": 260,
  "fbs": 0,
  "restecg": 0,
  "thalach": 145,
  "exang": 1,
  "oldpeak": 2.3,
  "slope": 1,
  "ca": 0,
  "thal": 1
}
```

**Prediction Response:**
```json
{
  "patient_id": null,
  "disease": "heart",
  "risk_score": 72.4,
  "calibrated_probability": 0.724,
  "risk_category": "HIGH",
  "confidence_interval": [0.68, 0.77],
  "top_features": [
    {"feature": "cp", "value": 2, "shap_value": 0.312, "direction": "increases_risk", "rank": 1},
    {"feature": "oldpeak", "value": 2.3, "shap_value": 0.187, "direction": "increases_risk", "rank": 2}
  ],
  "plain_english_summary": "This patient has HIGH heart disease risk (72.4%). Key factors: atypical chest pain and significant ST depression during exercise.",
  "clinical_action": {
    "action": "Specialist referral",
    "timeframe": "1 month",
    "urgency": "high"
  },
  "model_version": "heart_xgb_v1.2.0",
  "cached": false
}
```

### 5.3 Middleware Stack

1. **CORS** — Configurable allowed origins
2. **Rate Limiting** — 100 req/min general, 20 req/min predictions
3. **JWT Authentication** — Bearer token validation
4. **Request Logging** — Structured logs with correlation IDs

---

## 6. Frontend Dashboard

### 6.1 Page Structure

| Page | Components | Data Source |
|------|------------|-------------|
| **Login** | Animated form, brand showcase | `/auth/login` |
| **Dashboard** | KPI cards, trend charts, model status | `/analytics/summary` |
| **Predict** | Disease tabs, input forms, results display | `/predict/{disease}` |
| **Analytics** | Cluster scatter, risk distribution, SHAP importance | `/analytics/*` |
| **Patients** | Patient table, search, risk badges | `/patients/` |

### 6.2 State Management

```typescript
// React Query for server state
const { data, isLoading, isError } = useQuery({
    queryKey: ['analytics-summary'],
    queryFn: () => analyticsAPI.summary(),
    staleTime: 5 * 60 * 1000,  // 5 minutes
    retry: 2,
});

// Local state for forms
const [values, setValues] = useState<Record<string, number>>({});
```

### 6.3 Authentication Flow

1. User submits credentials → POST `/auth/login`
2. Tokens stored in localStorage (access + refresh)
3. Axios interceptor adds `Authorization: Bearer {token}` to all requests
4. On 401 response → Clear tokens, dispatch auth logout event
5. `AuthLogoutHandler` component listens for event, navigates to `/login`

---

## 7. Database Design

### 7.1 Entity Relationship Diagram

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│    User     │───1:N─│   Patient   │───1:N─│ Prediction  │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ id (UUID)   │       │ id (UUID)   │       │ id (UUID)   │
│ email       │       │ mrn         │       │ patient_id  │
│ hashed_pw   │       │ dob         │       │ disease     │
│ role        │       │ sex         │       │ probability │
│ is_active   │       │ clinician_id│       │ risk_cat    │
│ created_at  │       │ created_at  │       │ shap_json   │
└─────────────┘       └─────────────┘       └─────────────┘
                                                   │
                                            ┌──────▼──────┐
                                            │  AuditLog   │
                                            ├─────────────┤
                                            │ id (INT)    │
                                            │ user_id     │
                                            │ action      │
                                            │ resource    │
                                            │ ip_address  │
                                            │ timestamp   │
                                            └─────────────┘
```

### 7.2 Key Tables

**Users Table:**
- Stores clinicians, admins, researchers
- bcrypt-hashed passwords (cost factor 12)
- Role-based access control (RBAC)

**Patients Table:**
- MRN (Medical Record Number) as unique identifier
- Linked to primary clinician
- HIPAA-compliant data model

**Predictions Table:**
- Full input snapshot (JSONB)
- SHAP contributions (JSONB)
- Model version for reproducibility

---

## 8. Authentication & Security

### 8.1 JWT Token Structure

```json
{
  "sub": "user-uuid",
  "role": "clinician",
  "type": "access",
  "jti": "unique-token-id",
  "iat": 1711170000,
  "exp": 1711173600
}
```

### 8.2 Token Lifecycle

| Token Type | Lifetime | Storage |
|------------|----------|---------|
| Access Token | 60 minutes | localStorage |
| Refresh Token | 7 days | localStorage |

### 8.3 Security Measures

1. **Password Hashing** — bcrypt with cost factor 12
2. **Token Blacklisting** — Redis SET for revoked JTIs
3. **Rate Limiting** — slowapi with sliding window
4. **Input Validation** — Pydantic V2 with strict types
5. **CORS** — Configurable allowed origins
6. **SQL Injection** — Parameterized queries via SQLAlchemy ORM

### 8.4 RBAC Permissions

| Role | Permissions |
|------|-------------|
| superadmin | read, write, delete, admin, deploy |
| admin | read, write, delete, admin |
| clinician | read, write |
| patient | read |
| researcher | read |

---

## 9. Data Mining Features

### 9.1 Patient Clustering (GMM)

```python
# Gaussian Mixture Model for patient phenotyping
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4, covariance_type='full')
cluster_labels = gmm.fit_predict(patient_features)
```

**Clusters:**
- Cluster 0: Low-risk Healthy
- Cluster 1: Metabolic Syndrome
- Cluster 2: Cardiovascular High-risk
- Cluster 3: Multi-morbid Elderly

### 9.2 Association Rule Mining

```python
# FP-Growth for comorbidity patterns
from mlxtend.frequent_patterns import fpgrowth, association_rules
frequent_itemsets = fpgrowth(df_binary, min_support=0.1)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.5)
```

**Example Rules:**
- {diabetes} → {heart_disease} (confidence: 72%, lift: 2.4)
- {obesity, hypertension} → {diabetes} (confidence: 68%, lift: 2.1)

---

## 10. Deployment

### 10.1 Docker Compose Stack

```yaml
services:
  api:
    build: ./docker
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis]
  
  frontend:
    build: ./frontend
    ports: ["3000:80"]
  
  postgres:
    image: postgres:15
    volumes: [pgdata:/var/lib/postgresql/data]
  
  redis:
    image: redis:7-alpine
```

### 10.2 Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ENVIRONMENT` | development / production | `production` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `JWT_SECRET` | Token signing key (32+ chars) | `your-secret-key` |
| `MLFLOW_TRACKING_URI` | MLflow artifact location | `file:///app/mlruns` |

---

## 11. Testing Strategy

### 11.1 Test Structure

```
tests/
├── unit/
│   ├── test_feature_engineering.py
│   ├── test_risk_scorer.py
│   └── test_security.py
├── integration/
│   ├── test_auth_flow.py
│   └── test_prediction_endpoints.py
└── e2e/
    └── test_full_workflow.py
```

### 11.2 Coverage Target

- Minimum: 75% line coverage
- Critical paths: 100% (authentication, predictions)

### 11.3 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 12. Bug Fixes Applied

### 12.1 Security Fixes

| Issue | Fix |
|-------|-----|
| Hardcoded admin credentials | Implemented proper DB user authentication with bcrypt |
| Missing endpoint authentication | Added `Depends(get_current_user)` to all protected routes |
| Logout not invalidating tokens | Implemented Redis token blacklist |
| Credentials shown in frontend | Removed hardcoded values, improved error messages |

### 12.2 Error Handling Fixes

| Issue | Fix |
|-------|-----|
| Silent error swallowing | Added proper logging and error states |
| Generic error messages | Implemented specific HTTP status handling |
| Missing loading states | Added `isLoading` and `isError` to all queries |
| Race condition in Redis init | Added `asyncio.Lock` for thread-safe initialization |

### 12.3 Frontend Fixes

| Issue | Fix |
|-------|-----|
| `window.location.href` redirect | Changed to React Router `navigate()` |
| Index as React key | Changed to unique identifiers |
| Missing TypeScript types | Added proper type definitions |

---

## 13. Future Enhancements

### 13.1 Short-term (Phase 2)

- [ ] PDF report generation with ReportLab
- [ ] Real-time model retraining pipeline
- [ ] Mobile-responsive layout improvements
- [ ] Patient import/export (CSV)

### 13.2 Medium-term (Phase 3)

- [ ] Multi-language support (i18n)
- [ ] Real-time WebSocket notifications
- [ ] Federated learning for privacy-preserving updates
- [ ] FHIR integration for EHR interoperability

### 13.3 Long-term

- [ ] Survival analysis models (time-to-event)
- [ ] Longitudinal risk tracking (velocity trends)
- [ ] Multi-disease ensemble predictions
- [ ] Edge deployment (TensorFlow Lite / ONNX)

---

## Appendix A: API Error Codes

| Code | Description | Action |
|------|-------------|--------|
| 401 | Unauthorized | Re-authenticate |
| 403 | Forbidden | Check role permissions |
| 404 | Not Found | Verify resource exists |
| 422 | Validation Error | Check request payload |
| 429 | Rate Limited | Wait and retry |
| 503 | Model Not Loaded | Retry after startup |

---

## Appendix B: Feature Reference

### Heart Disease Features

| Feature | Description | Range |
|---------|-------------|-------|
| age | Age in years | 20-90 |
| sex | 1=Male, 0=Female | 0-1 |
| cp | Chest pain type | 0-3 |
| trestbps | Resting blood pressure (mmHg) | 80-220 |
| chol | Serum cholesterol (mg/dl) | 100-600 |
| fbs | Fasting blood sugar > 120 | 0-1 |
| restecg | Resting ECG results | 0-2 |
| thalach | Maximum heart rate | 60-220 |
| exang | Exercise induced angina | 0-1 |
| oldpeak | ST depression | 0-7 |
| slope | ST segment slope | 0-2 |
| ca | Major vessels colored | 0-3 |
| thal | Thalassemia type | 0-2 |

---

*Report generated: March 2026*
*Version: 1.0.0*
