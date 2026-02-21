# MedPredict — Complete Project Documentation

> **Version:** 1.0.0 | **Last Updated:** February 2026 | **Stack:** Python 3.11 · FastAPI · React 18 · XGBoost · MLflow

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Live Application Screenshots](#2-live-application-screenshots)
3. [System Architecture](#3-system-architecture)
4. [Frontend — React Application](#4-frontend--react-application)
5. [Backend — FastAPI Application](#5-backend--fastapi-application)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Training — Optuna + Nested CV](#8-model-training--optuna--nested-cv)
9. [Risk Scoring Engine](#9-risk-scoring-engine)
10. [Explainability — SHAP](#10-explainability--shap)
11. [Authentication & Authorization](#11-authentication--authorization)
12. [Database Schema — PostgreSQL + Alembic](#12-database-schema--postgresql--alembic)
13. [Caching — Redis](#13-caching--redis)
14. [Docker Deployment](#14-docker-deployment)
15. [CI/CD Pipeline — GitHub Actions](#15-cicd-pipeline--github-actions)
16. [Data Sources](#16-data-sources)
17. [API Schemas Reference](#17-api-schemas-reference)
18. [Configuration Reference](#18-configuration-reference)
19. [Running Everything Locally](#19-running-everything-locally)
20. [Glossary](#20-glossary)

---

## 1. Project Overview

**MedPredict** is a production-grade multi-disease risk stratification platform designed for clinical use. It allows medical professionals to enter a patient's  lab values and demographic information and instantly receive a machine-learning-generated probability of disease — with a clinical risk category, a confidence interval, a SHAP-based explanation of which biomarkers drove the prediction, and a recommended clinical action.

### Core Goals

| Goal | Implementation |
|---|---|
| Accurate disease risk prediction | Optuna-tuned XGBoost / LightGBM with calibrated probabilities |
| Explainable predictions | SHAP TreeExplainer showing top-5 feature contributions |
| Clinical safety | Conservative cancer thresholds (HIGH sensitivity), Wilson CI intervals |
| Production readiness | JWT auth, Redis cache, PostgreSQL persistence, Docker |
| Clinician UX | Clean glassmorphism React dashboard with per-disease tabs |

### Diseases Covered

**Heart Disease** — Predicts likelihood of significant coronary artery disease using 13 standard ACC/AHA-aligned biomarkers from the Cleveland Heart Disease dataset.

**Type 2 Diabetes** — Predicts diabetes onset using 8 metabolic markers from the PIMA Indian Diabetes Study, extended with feature-engineered HOMA-IR proxy, obesity flags, and glucose thresholds per ADA 2024 guidelines.

**Breast Cancer** — Binary classification (Malignant / Benign) using all 30 Wisconsin Diagnostic Breast Cancer (WDBC) features: mean, SE, and worst-case measurements of 10 cell nucleus properties.

**Chronic Kidney Disease (CKD)** — Predicts CKD risk from 20 clinical and lab features including eGFR proxy (CKD-EPI formula), creatinine, hemoglobin, blood pressure, and comorbidity flags.

---

## 2. Live Application Screenshots

> The frontend is served by Vite at `http://localhost:5173`. The API runs at `http://localhost:8000`.

### Dashboard — Population Overview

The Dashboard gives clinicians a bird's-eye view of the patient population's risk profile.

![Dashboard](file:///Users/aryantiwari/.gemini/antigravity/brain/c4eb3a11-a1ff-4c11-a343-f02f31c7f29a/dashboard_page_1771697026419.png)

**What you see:**
- **Strategic Metric Cards** (top row): Total Predictions made, High-Risk Patients count, Heart Disease case count, Diabetes case count — updated in real time
- **Prediction Trend Chart** (bottom left): A Recharts area chart showing prediction volume broken down by day over the last 7 days. Green = total predictions, Red = high-risk predictions. Useful for spotting sudden spikes in risk referrals.
- **Disease Risk Radar** (bottom right): A hexagonal Recharts radar chart plotting the average risk score (%) for each of the four diseases. Allows clinicians to identify if one disease category is trending higher than others at a population level.
- **Disease Breakdown Table**: A tabular summary showing each disease's total prediction count, average risk score, and operational status (Active/Inactive).

---

### Predict — Real-Time Risk Assessment

The Predict page is the core clinical tool, where a clinician enters a single patient's biomarkers and receives an AI assessment.

![Predict Page](file:///Users/aryantiwari/.gemini/antigravity/brain/c4eb3a11-a1ff-4c11-a343-f02f31c7f29a/predict_page_1771697077763.png)

**How it works:**
1. The clinician selects a disease tab (Heart Disease | Diabetes | Cancer | Kidney)
2. They enter the patient's biomarkers in the labeled form fields — each field has a tooltip with valid ranges
3. Clicking **Run Prediction** sends a POST request to `/api/v1/predict/<disease>` with the values as JSON
4. The right panel updates with the AI assessment result:
   - **Calibrated Probability** (0–100%): The model's calibrated output after isotonic regression — a well-calibrated 70% means the condition is present in ~70% of similar patients historically
   - **Risk Category**: LOW / BORDERLINE / MODERATE / HIGH / CRITICAL — determined by disease-specific thresholds (more sensitive for cancer)
   - **Confidence Interval**: A 95% Wilson interval around the probability estimate
   - **Top Features**: SHAP-derived top 5 biomarkers driving the prediction (increases/decreases risk, with magnitude)
   - **Plain-English Summary**: Auto-generated natural language explanation (e.g., "Predicted heart risk: 27% (BORDERLINE). Primary risk drivers: ST depression, age, chest pain type.")
   - **Clinical Action**: Recommended next step with urgency and timeframe (e.g., "Specialist referral, 1 month")

---

### Analytics — Population Phenotyping

![Analytics Page](file:///Users/aryantiwari/.gemini/antigravity/brain/c4eb3a11-a1ff-4c11-a343-f02f31c7f29a/analytics_page_1771697087275.png)

The Analytics page moves from per-patient inference to population-level insights:

- **Patient Phenotype Clusters**: PCA + Gaussian Mixture Model clustering projected onto 2D. Each point is a patient, colored by cluster. This identifies natural subgroups — e.g., "young low-risk" vs "elderly multi-morbid" — without manual segmentation.
- **Population Risk Distribution**: Bar charts showing what fraction of the entire patient population falls into each risk category for each disease. Helps hospital administration understand resource demand.
- **Comorbidity Association Rules**: FP-Growth mining over patient histories to surface rules like "Heart Disease + Diabetes → 3× more likely to have CKD" with confidence and lift metrics. These are surfaced as cards with rule text, confidence, and lift displayed.

---

### Patients — Patient Registry

![Patients Page](file:///Users/aryantiwari/.gemini/antigravity/brain/c4eb3a11-a1ff-4c11-a343-f02f31c7f29a/patients_page_1771697097263.png)

A searchable table of all registered patients with:
- Name, ID, date of birth, last assessment date
- Latest risk categories per disease (color-coded chips: green=LOW, red=HIGH)
- Click-through to full patient history page with all past predictions and trends
- Add New Patient form integrated into the sidebar

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   FRONTEND — React 18 / Vite                        │
│                                                                     │
│  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌──────────────────┐   │
│  │  Login   │  │Dashboard│  │  Predict  │  │Analytics Patients│   │
│  └──────────┘  └─────────┘  └───────────┘  └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ REST API (JSON) over HTTP
                            │ Bearer JWT in Authorization header
┌───────────────────────────▼─────────────────────────────────────────┐
│                   BACKEND — FastAPI / Python 3.11                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Uvicorn ASGI server                                         │  │
│  │  ┌─────────────┐   ┌─────────────────────────────────────┐  │  │
│  │  │  Middleware  │   │  Routers                            │  │  │
│  │  │  - CORS      │   │  /auth  /predict  /analytics        │  │  │
│  │  │  - Rate limit│   │  /patients  /reports  /health       │  │  │
│  │  └─────────────┘   └─────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │ PredictionEngine│  │  RiskScorer   │  │   DiseaseExplainer     │ │
│  │  - Load MLflow│  │  - Composite  │  │   - SHAP TreeExplainer │ │
│  │  - SHAP setup │  │    score       │  │   - Top-5 features     │ │
│  │  - Inference  │  │  - Wilson CI   │  │   - Plain English      │ │
│  └──────────────┘  └────────────────┘  └────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────┘
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    ┌─────▼──────┐   ┌──────▼──────┐   ┌─────▼──────┐
    │  MLflow     │   │  PostgreSQL │   │   Redis    │
    │  mlruns/    │   │  + Alembic  │   │  (cache)   │
    │  (models)   │   │  migrations │   │  TTL=5min  │
    └─────────────┘   └────────────┘   └────────────┘
```

### Request Lifecycle (Prediction)

```
Client → POST /api/v1/predict/heart
      → JWT middleware validates token
      → Rate limiter checks (20 req/min)
      → Router calls _run_prediction("heart", data)
        → Cache key computed (SHA-256 of sorted features)
        → Redis MISS → proceed
        → PredictionEngine.predict("heart", df)
          → run_in_executor (thread pool, CPU-bound)
            → apply_feature_engineering("heart", df)
            → pipeline.predict_proba(df)[:, 1]   ← calibrated model
            → SHAP TreeExplainer.shap_values(X_preprocessed)
        → RiskScorer.compute(prob)
          → composite = 0.7×prob + 0.2×velocity + 0.1×comorbidity
          → Wilson CI
          → categorize → clinical action
        → Build PredictionResponse
        → Redis SET (TTL=300s)
      → JSON response
```

---

## 4. Frontend — React Application

**Location:** `frontend/`  
**Technology:** React 18, TypeScript, Vite, Recharts, CSS custom properties

### Pages and Components

#### Login Page (`frontend/src/pages/Login.tsx`)

A clean full-screen login form with:
- Email + password inputs with validation
- Demo credential hint (`admin@example.com / admin`)
- Calls `POST /api/v1/auth/login`
- On success: stores `access_token` in memory + `refresh_token` in HttpOnly cookie (configurable)
- Redirects to Dashboard

#### Dashboard Page (`frontend/src/pages/Dashboard.tsx`)

Fetches aggregate data from `GET /api/v1/analytics/summary` and renders:

- **MetricCard components**: Each shows an icon, label, count, and a trend indicator (e.g., "+12% this week"). The high-risk card turns red when count > threshold.
- **PredictionTrendChart**: A Recharts `AreaChart` with two series (total/high-risk). X-axis = date (last 7 days), Y-axis = count. Gradient fill for visual appeal.
- **DiseaseRiskRadar**: A Recharts `RadarChart` with 4 axes (Heart / Diabetes / Cancer / Kidney). Each axis value = average population risk score. Useful for spotting disease trends.
- **DiseaseBreakdownTable**: Scrollable table with sortable columns.

#### Predict Page (`frontend/src/pages/Predict.tsx`)

The most feature-rich page — handles 4 different disease forms via tabs:

- **Tab switching**: Each tab (Heart | Diabetes | Cancer | Kidney) loads a different `FormSchema` — the set of fields and their validation rules are disease-specific.
- **Form validation**: Real-time range validation on all numeric fields. Out-of-range values highlighted in red with a tooltip showing valid range.
- **API call**: `POST /api/v1/predict/{disease}` with a JSON body built from the form state.
- **Result panel** (right side, visible after "Run Prediction"):
   - Large probability gauge with color-coded fill (green → yellow → red)
   - Risk badge (chip) with category text
   - CI bar showing [lower, upper] bounds
   - Feature contributions list (top 5 SHAP values, sorted by magnitude)
   - Each contribution shows: feature name, patient value, SHAP direction arrow, magnitude
   - Clinical action box with action text, timeframe, urgency color

#### Analytics Page (`frontend/src/pages/Analytics.tsx`)

Fetches from two endpoints:

1. `GET /api/v1/analytics/clusters` → scatter plot (PCA x/y, cluster ID, patient ID)
2. `GET /api/v1/analytics/comorbidities` → association rule cards

Renders:
- **Cluster Scatter Plot**: Recharts `ScatterChart`. Points colored by cluster ID. Hover shows patient ID + risk score.
- **Risk Distribution Bars**: Stacked bar chart showing % of patients in each of 5 risk categories, broken down by disease. Useful for comparing population health across disease types.
- **Comorbidity Rules**: Cards each showing the antecedent→consequent rule, confidence (%), and lift ratio. Cards sorted by lift descending.

#### Patients Page (`frontend/src/pages/Patients.tsx`)

Two sub-views:

1. **Patient List**: Searchable, paginated table. Columns: ID, Name, DOB, Last Visit, and risk chips for each disease. Click a row to open the patient detail panel.
2. **Patient Detail Panel** (slide-in): Shows all past predictions for the selected patient in a timeline. Each past prediction shows disease, date, probability, and risk category. Trend sparklines show how a patient's risk has evolved over time.

---

## 5. Backend — FastAPI Application

**Entry Point:** `src/api/main.py`

### Application Factory

The `create_app()` function creates a FastAPI instance configured with:

- **CORS middleware**: Allows origins from `settings.cors_origins` (configurable via `.env`)
- **SlowAPI rate limiting**: 100 req/min general, 20 req/min for prediction endpoints
- **Lifespan context**: At startup, calls `await prediction_engine.preload_models()` to load all 4 ML models into memory from MLflow, ensuring zero cold-start latency for the first prediction request.
- **Router mounting**: Each feature domain has its own router with a `/api/v1/{domain}` prefix.

### Dependency Injection

`src/api/deps.py` provides three shared dependencies:

| Dependency | What it provides |
|---|---|
| `get_current_user()` | Decodes JWT, validates type=access, returns `{user_id, role, jti}` |
| `require_permission(perm)` | Factory: returns a dependency that checks RBAC for the given permission |
| `get_redis()` | Returns async Redis connection pool, or a no-op in-memory stub if Redis unavailable |

### Prediction Router (`src/api/routers/predict.py`)

Implements the `PredictionEngine` singleton and all 4 prediction endpoints.

**PredictionEngine** is the central orchestrator:

```python
class PredictionEngine:
    _models: dict[str, Any]       # disease → fitted CalibratedClassifierCV pipeline
    _explainers: dict[str, Any]   # disease → SHAP TreeExplainer
    _feature_names: dict[str, list[str]]  # disease → ordered feature names post-preprocessing
    _background_data: dict[str, pd.DataFrame]  # disease → 50-row background for SHAP
```

**`preload_models()`** — runs at startup in a thread pool:
1. Detects if MLflow URI points to a running server or local file store
2. Attempts to load each disease model using alias cascade: `@latest` → `/1` → `/2`
3. After loading models, calls `_setup_shap_explainers()`:
   - Loads the training dataset for each disease (via `src/data/load.py`)
   - Draws a 50-row background sample
   - Applies feature engineering
   - Extracts the inner tree model from inside the `CalibratedClassifierCV` wrapper
   - Initializes a `shap.TreeExplainer` with `feature_perturbation="tree_path_dependent"`

**`predict(disease, df)`** — async inference:
1. Offloads CPU work to `ThreadPoolExecutor` via `asyncio.run_in_executor`
2. Applies feature engineering to the input DataFrame
3. Runs `model.predict_proba(X)[:, 1]` for the calibrated probability
4. Transforms through the preprocessor and runs SHAP for feature contributions

**Cache flow**: Every prediction is keyed by SHA-256 of the sorted feature values. On cache hit, returns immediately (sub-millisecond). On miss, runs inference, stores result with 5-minute TTL.

---

## 6. Machine Learning Pipeline

**Location:** `src/features/pipeline.py`

The preprocessing pipeline is built using `imbalanced-learn`'s `ImbPipeline`, which supports SMOTE as a pipeline step. This is critical for correctness — SMOTE must only be applied to training folds, never to validation folds, which vanilla sklearn `Pipeline` does not enforce.

### Pipeline Stages (in order)

```
Input DataFrame
    │
    ▼  
ColumnTransformer
    ├── Numeric branch
    │       ├── IterativeImputer  ← MICE-style multivariate imputation
    │       └── RobustScaler      ← IQR-based scaling (handles outliers)
    │
    └── Categorical branch
            ├── SimpleImputer(strategy='most_frequent')
            └── OneHotEncoder(drop='first', handle_unknown='ignore')
    │
    ▼
[SMOTE or BorderlineSMOTE]   ← only in training, only when imbalance ratio > threshold
    │
    ▼
Classifier  ← XGBoost / LightGBM / RandomForest / LogisticRegression
```

### Disease-Specific Configuration (`FEATURE_CONFIG`)

Each disease has its own feature configuration:

**Heart Disease:**
- Numeric: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- Categorical: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`
- SMOTE: Yes (imbalance ratio threshold = 3:1)

**Diabetes:**
- Numeric: All 8 PIMA features (pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age)
- Categorical: None
- SMOTE: Yes (imbalance ratio threshold = 2:1, as PIMA is ~35% positive)

**Cancer:**
- Numeric: All 30 WDBC features (auto-detected)
- Categorical: None
- SMOTE: Disabled (57% / 43% split is relatively balanced)

**Kidney Disease:**
- Numeric: 14 clinical lab values (age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc)
- Categorical: 6 comorbidity flags (htn, dm, cad, appet, pe, ane)
- SMOTE: Disabled

### Why IterativeImputer (MICE)?

Medical datasets frequently have missing values that are **Missing At Random (MAR)** — e.g., insulin values are often not measured in patients without diabetes symptoms. MICE (Multiple Imputation by Chained Equations) imputes each missing value by fitting a regression model using all other features as predictors. This preserves inter-feature correlations far better than mean imputation, which is critical for tree-based models that already split on feature interactions.

### Why RobustScaler?

Clinical data contains biologically determined outliers that should not be removed — a cholesterol of 550 is extreme but clinically meaningful. RobustScaler uses the IQR (25th–75th percentile) instead of mean/std for centering and scaling, so outliers do not distort the distribution for the majority of patients.

### Why BorderlineSMOTE vs SMOTE?

Standard SMOTE synthesizes new minority-class examples uniformly between two minority points. **BorderlineSMOTE** only synthesizes between minority points near the decision boundary — points that are at risk of being misclassified. This produces a harder training set that generalizes better, particularly for heart disease where the boundary between positive and negative cases is clinically subtle.

---

## 7. Feature Engineering

**Location:** `src/features/engineering.py`

All engineered features are clinically motivated — each has an explicit citation to clinical guidelines or epidemiological literature.

### Heart Disease Features

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `age_chol_ratio` | `chol / age` | ACC/AHA 2019: age-cholesterol synergy increases risk super-linearly |
| `hr_age_product` | `thalach × age` | Duke Treadmill Score proxy for exercise capacity |
| `chol_high` | `chol > 240` (binary) | AHA borderline-high cholesterol threshold |
| `chol_very_high` | `chol > 280` (binary) | AHA high-risk cholesterol threshold |
| `tachycardia_flag` | `thalach > 100` (binary) | Tachycardia at rest is an independent risk factor |
| `hypertension_flag` | `trestbps > 130` (binary) | AHA 2017 Stage 1 hypertension threshold |
| `ischemia_severity` | `oldpeak × exang` | ST depression combined with exercise-induced angina = ischemia indicator |

### Diabetes Features

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `obese` | `BMI ≥ 30` | WHO obesity classification (binary) |
| `severely_obese` | `BMI ≥ 35` | WHO Class II obesity — significantly elevated diabetes risk |
| `glucose_prediabetic` | `glucose ∈ [100, 125]` | ADA 2024 prediabetes range (IFG) |
| `glucose_diabetic` | `glucose > 126` | ADA 2024 diabetes diagnostic threshold |
| `homa_ir_proxy` | `(glucose × insulin) / 405` | HOMA-IR: insulin resistance proxy (using mg/dL units) |
| `glucose_bmi` | `glucose × bmi` | Metabolic syndrome risk interaction |
| `preg_bmi` | `pregnancies × bmi` | Reproductive-metabolic burden: gestational diabetes risk proxy |
| `glucose_age_ratio` | `glucose / age` | Glucose intolerance increases with age — ratio captures this |

### Breast Cancer Features

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `{feature}_deterioration` | `worst / mean` for radius, perimeter, area, concavity, concave_points | Captures progressive cellular deterioration between mean and worst-case measurements |
| `shape_irregularity` | `perimeter² / (4π × area)` | Isoperimetric quotient: value > 1 indicates non-circular (irregular) shape, a hallmark of malignancy |

### Kidney Disease Features

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `egfr_proxy` | `186 × creatinine^(-1.154) × age^(-0.203)` | CKD-EPI simplified eGFR formula (the gold standard for kidney staging) |
| `ckd_stage_proxy` | `pd.cut(egfr_proxy, bins=[0,15,30,45,60,90,∞])` | Maps eGFR to CKD Stages 5→0 per KDIGO 2012 guidelines |
| `anemia_flag` | `hemoglobin < 12.0` | Anemia is a direct consequence of erythropoietin deficiency in CKD |
| `hypertension_severe` | `bp > 90 (diastolic)` | Severe hypertension accelerates CKD progression |

---

## 8. Model Training — Optuna + Nested CV

**Location:** `src/models/train.py`  
**Entry Point:** `scripts/run_training.py`

### Overview

The training framework implements **nested cross-validation** with **Optuna** hyperparameter optimization — the gold standard for avoiding optimistic bias in model evaluation.

```
Outer CV (5-fold)
  ├── Fold 1: test on 20% of data
  │     Inner CV (Optuna, 100 trials, 3-fold)
  │       └── Searches XGBoost / LightGBM hyperparameters
  │     Best inner params → fit on 80%, evaluate on test 20%
  ├── Fold 2 ... Fold 5 (same process)
  └── Average outer metrics → unbiased performance estimate

Final Model: Optuna on full dataset → CalibratedClassifierCV (cv=5) → MLflow register
```

### Hyperparameter Search Space

For **XGBoost**:
- `n_estimators`: 50–500
- `max_depth`: 3–8
- `learning_rate`: 0.005–0.3 (log-scale)
- `subsample`: 0.5–1.0
- `colsample_bytree`: 0.5–1.0
- `reg_alpha` (L1): 1e-8 to 1.0 (log-scale)
- `reg_lambda` (L2): 1e-8 to 1.0 (log-scale)

For **LightGBM** (diabetes):
- `num_leaves`: 20–300
- `min_child_samples`: 5–100
- `feature_fraction`: 0.4–1.0
- `bagging_fraction`: 0.4–1.0
- `learning_rate`: 0.005–0.3

### Evaluation Metrics

The primary metric is **AUC-PR (Area Under Precision-Recall Curve)** rather than AUC-ROC. This is because the medical datasets are class-imbalanced (e.g., only 35% of PIMA subjects have diabetes). AUC-ROC is optimistic under imbalance; AUC-PR better reflects performance on the minority class (disease-positive patients).

Secondary metrics tracked:
- AUC-ROC (overall discrimination)
- Sensitivity (Recall for positive class) — critical for cancer
- Specificity (Recall for negative class)
- Brier Score (calibration quality)

### Probability Calibration

Raw XGBoost/LightGBM `.predict_proba()` outputs are not reliably calibrated — a model's "70%" may not correspond to a 70% actual disease prevalence. We apply **isotonic regression calibration** via `CalibratedClassifierCV(cv=5, method='isotonic')`.

This fits 5 calibrators in cross-validation on the training data, ensuring the calibration itself is not overfit. The result: a probability score that is genuinely interpretable as a disease prevalence estimate.

### MLflow Tracking

Every training run logs:
- All hyperparameters (Optuna best trial)
- All evaluation metrics (inner + outer CV averages)
- The fitted pipeline artifact (serialized via `mlflow.sklearn.log_model`)
- A model signature (input schema) for validation at inference
- Tags: disease name, algorithm, training date, dataset version

After training, each model is registered in the **MLflow Model Registry** as `disease-prediction-{disease}` version 1+.

---

## 9. Risk Scoring Engine

**Location:** `src/scoring/risk_scorer.py`

### Composite Score Formula

The raw model probability is not directly shown to clinicians — it is normalized through a composite score that accounts for three dimensions:

```
composite = ALPHA × calibrated_probability
          + BETA  × velocity           (change from last visit)
          + GAMMA × comorbidity_index  (burden of other conditions)

where:
  ALPHA = 0.70   (primary signal)
  BETA  = 0.20   (trajectory matters)
  GAMMA = 0.10   (burden matters)

velocity is clamped to [-0.3, +0.3] to prevent outlier trajectories from dominating
```

**Example:** A patient with 60% heart probability (MODERATE), whose risk increased from 40% last visit (velocity = +0.20), and who has diabetes (comorbidity_index = 0.2):
```
composite = 0.70×0.60 + 0.20×0.20 + 0.10×0.20
          = 0.42 + 0.04 + 0.02
          = 0.48 → MODERATE risk
```

### Disease-Specific Risk Thresholds

Risk category boundaries are set per-disease, calibrated to maximize **Youden's J statistic** on validation cohorts:

| Category | Heart | Diabetes | Cancer |Kidney |
|---|---|---|---|---|
| LOW | < 15% | < 20% | < 10% | < 15% |
| BORDERLINE | 15–30% | 20–35% | 10–25% | 15–30% |
| MODERATE | 30–55% | 35–55% | 25–50% | 30–55% |
| HIGH | 55–75% | 55–75% | 50–65% | 55–75% |
| CRITICAL | > 75% | > 75% | > 65% | > 75% |

Note: Cancer thresholds are more conservative (lower cutoffs) — missing a cancer case is far more harmful than a false positive.

### Clinical Actions by Category

| Category | Action | Urgency | Timeframe |
|---|---|---|---|
| LOW | Routine screening | Low | 12–24 months |
| BORDERLINE | Lifestyle modification counseling | Low | 6 months |
| MODERATE | Clinical review + additional labs | Medium | 3 months |
| HIGH | Specialist referral | High | 1 month |
| CRITICAL | Emergency clinical evaluation | Urgent | 24–48 hours |

### Confidence Intervals

The 95% confidence interval uses the **Wilson score interval** for proportions, which outperforms the naive ± 2σ interval at extreme probabilities (near 0 or 1). The CI is computed analytically from the calibrated probability — no bootstrap needed, making it zero additional computation cost at inference time.

---

## 10. Explainability — SHAP

**Location:** `src/explainability/shap_explainer.py`

### Why SHAP?

SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory and provides feature attributions with mathematically guaranteed properties:
- **Efficiency**: SHAP values sum to the difference between model output and base value
- **Consistency**: Removing a feature never increases its attribution
- **Null player**: A feature with no model impact gets SHAP value = 0

This is critical for clinical settings, where the model's reasoning must be auditable and legally defensible.

### Implementation

`DiseaseExplainer` uses **SHAP TreeExplainer**, which exploits the tree structure of XGBoost/LightGBM to compute exact SHAP values in O(TLD) time (T=trees, L=leaves, D=depth) — suitable for real-time inference.

```python
explainer = shap.TreeExplainer(
    model=inner_tree_model,
    data=background_df_50_rows,
    feature_perturbation="tree_path_dependent",
)
```

`feature_perturbation="tree_path_dependent"` is preferred over "interventional" when features are correlated (which they are in clinical data — age and cholesterol, BMI and glucose, etc.) because it computes conditional expectations respecting the tree structure, avoiding extrapolation into implausible feature combinations.

### Feature Contributions Output

```json
"top_features": [
  { "feature": "oldpeak", "value": 2.3, "shap_value": 0.18, "direction": "increases_risk", "rank": 1 },
  { "feature": "age", "value": 55, "shap_value": 0.09, "direction": "increases_risk", "rank": 2 },
  { "feature": "thalach", "value": 150, "shap_value": -0.06, "direction": "decreases_risk", "rank": 3 }
]
```

Each element shows:
- **feature**: the clinical variable name
- **value**: the patient's actual value for that variable
- **shap_value**: how much this feature's value shifted the probability up (positive) or down (negative)
- **direction**: "increases_risk" | "decreases_risk" | "neutral"
- **rank**: 1 = biggest contributor

---

## 11. Authentication & Authorization

**Location:** `src/auth/security.py`

### JWT Flow

```
POST /api/v1/auth/login  { email, password }
  → BCrypt.verify(password, hashed_password)
  → create_token(user_id, role, type="access", expires=30min)
  → create_token(user_id, role, type="refresh", expires=7days)
  → Return { access_token, refresh_token, token_type: "bearer" }

POST /api/v1/auth/refresh  { refresh_token }
  → decode_token(refresh_token)
  → Verify type="refresh"
  → Issue new access_token
```

Access tokens expire in 30 minutes. Refresh tokens expire in 7 days. Both are signed with HS256 (HMAC-SHA256) using `JWT_SECRET` from settings.

### Role-Based Access Control (RBAC)

| Permission | admin | clinician | viewer |
|---|---|---|---|
| predict | ✅ | ✅ | ❌ |
| read_patients | ✅ | ✅ | ✅ |
| write_patients | ✅ | ✅ | ❌ |
| read_analytics | ✅ | ✅ | ✅ |
| delete | ✅ | ❌ | ❌ |
| manage_users | ✅ | ❌ | ❌ |

---

## 12. Database Schema — PostgreSQL + Alembic

**Location:** `src/db/models.py`, `alembic/`

### Tables

```sql
-- Users for authentication
users (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email      VARCHAR(255) UNIQUE NOT NULL,
  full_name  VARCHAR(255),
  hashed_password TEXT NOT NULL,
  role       VARCHAR(50) DEFAULT 'viewer',   -- admin / clinician / viewer
  is_active  BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT now()
)

-- Patient registry
patients (
  id          UUID PRIMARY KEY,
  external_id VARCHAR(100) UNIQUE,              -- hospital MRN
  full_name   VARCHAR(255) NOT NULL,
  date_of_birth DATE,
  sex         VARCHAR(10),
  created_by  UUID REFERENCES users(id),
  created_at  TIMESTAMP DEFAULT now()
)

-- Individual prediction records
predictions (
  id               UUID PRIMARY KEY,
  patient_id       UUID REFERENCES patients(id) ON DELETE CASCADE,
  disease          VARCHAR(50) NOT NULL,         -- heart / diabetes / cancer / kidney
  input_features   JSONB NOT NULL,               -- raw biomarker values
  calibrated_prob  FLOAT NOT NULL,
  composite_score  FLOAT NOT NULL,
  risk_category    VARCHAR(50) NOT NULL,
  shap_values      JSONB,                        -- top feature contributions
  model_version    VARCHAR(100),
  created_by       UUID REFERENCES users(id),
  created_at       TIMESTAMP DEFAULT now()
)

-- Immutable audit trail (append-only)
audit_logs (
  id          BIGSERIAL PRIMARY KEY,
  user_id     UUID REFERENCES users(id),
  action      VARCHAR(100),                      -- login / predict / delete_patient etc.
  resource    VARCHAR(100),
  detail      JSONB,
  ip_address  INET,
  created_at  TIMESTAMP DEFAULT now()
)
```

### Running Migrations

```bash
# Create a new migration (after changing models.py)
alembic revision --autogenerate -m "description"

# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

---

## 13. Caching — Redis

Every prediction is cached in Redis with a 5-minute TTL. This dramatically reduces repeated API calls for the same patient (e.g., when a clinician refreshes the result page).

**Cache key**: `predict:{disease}:{SHA256[:16](sorted_features)}`

This means two different patients with the same features will share a cached result — clinically correct, since the model output is deterministic given features.

**Fallback**: If Redis is not running (common in local development), `deps.py` catches the connection error on the first call and provides a `_NoOpRedis` stub that implements `get()` (returns None) and `setex()` (no-op). The prediction endpoint continues to function correctly — just without caching. The fallback is transparent to the caller.

---

## 14. Docker Deployment

**Location:** `docker/Dockerfile`, `docker/docker-compose.yml`

### Multi-Stage Dockerfile

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[prod]"

# Stage 2: Production image (smaller)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY src/ ./src/
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Services

```yaml
services:
  api:          # FastAPI backend
  frontend:     # Nginx serving React build
  postgres:     # PostgreSQL 15
  redis:        # Redis 7 for caching
  mlflow:       # MLflow tracking server (port 5001)
  nginx:        # Reverse proxy (443 → 8000, 3000)
```

Start everything:
```bash
docker compose -f docker/docker-compose.yml up -d
```

---

## 15. CI/CD Pipeline — GitHub Actions

**Location:** `.github/workflows/ci.yml`

On every push and pull request to `main`:

1. **Lint**: `ruff check src/ tests/`
2. **Type check**: `mypy src/`
3. **Unit tests**: `pytest tests/unit/ -v`
4. **Integration tests**: `pytest tests/integration/ -v` (with PostgreSQL service container)
5. **Build Docker image**: `docker build -f docker/Dockerfile .`
6. **Push to registry** (on `main` branch only): `docker push ghcr.io/...`

---

## 16. Data Sources

| Disease | Dataset | Source | Rows | License |
|---|---|---|---|---|
| Heart Disease | Cleveland Heart Disease | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | CC BY 4.0 |
| Diabetes | PIMA Indian Diabetes | [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | CC0 Public Domain |
| Breast Cancer | WDBC | `sklearn.datasets.load_breast_cancer()` | 569 | BSD |
| Kidney Disease | CKD Synthetic | UCI format, synthetically generated | 400 | — |

---

## 17. API Schemas Reference

### Prediction Request — Heart Disease

```json
{
  "age": 55,           // float, 1–120
  "sex": 1,            // int, 0=Female 1=Male
  "cp": 2,             // int, 0–3 (chest pain type)
  "trestbps": 145,     // float, mmHg (resting BP)
  "chol": 233,         // float, mg/dL (serum cholesterol)
  "fbs": 1,            // int, 0/1 (fasting blood sugar > 120)
  "restecg": 0,        // int, 0–2 (resting ECG)
  "thalach": 150,      // float (max heart rate)
  "exang": 0,          // int, 0/1 (exercise-induced angina)
  "oldpeak": 2.3,      // float (ST depression)
  "slope": 0,          // int, 0–2 (ST slope)
  "ca": 0,             // int, 0–3 (major vessels)
  "thal": 1,           // int, 0–2 (thalassemia)
  "patient_id": "P-001" // optional string
}
```

### Prediction Response (all diseases)

```json
{
  "patient_id": "P-001",
  "disease": "heart",
  "risk_score": 0.188,                     // composite score ∈ [0, 1]
  "calibrated_probability": 0.268,         // model calibrated output
  "risk_category": "BORDERLINE",
  "confidence_interval": [0.229, 0.307],   // 95% Wilson CI
  "velocity": null,                        // change from previous prediction
  "top_features": [
    {
      "feature": "oldpeak",
      "value": 2.3,
      "shap_value": 0.18,
      "direction": "increases_risk",
      "rank": 1
    }
  ],
  "plain_english_summary": "Predicted heart risk: 27% (BORDERLINE). Primary risk drivers: ST depression, age.",
  "clinical_action": {
    "action": "Lifestyle modification counseling",
    "timeframe": "6 months follow-up",
    "urgency": "low"
  },
  "model_version": "heart_v1.0",
  "cached": false
}
```

---

## 18. Configuration Reference

All settings are in `src/config.py` (Pydantic Settings) and loaded from `.env`:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Enables debug logs, disables some security checks |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `MLFLOW_TRACKING_URI` | `file:///…/mlruns` | MLflow server or local file store |
| `DATABASE_URL` | PostgreSQL DSN | Full async-compatible DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `JWT_SECRET` | *(required, ≥32 chars)* | HMAC signing key for tokens |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh token lifetime |
| `PREDICTION_CACHE_TTL_SECONDS` | `300` | Redis cache TTL (5 minutes) |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | Allowed frontend origins |
| `RATE_LIMIT_GENERAL` | `100/minute` | Global rate limit |
| `RATE_LIMIT_PREDICTION` | `20/minute` | Prediction endpoint rate limit |

---

## 19. Running Everything Locally

### Step 1: Set Up Python Environment

```bash
cd "Disease Prediction"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env — set MLFLOW_TRACKING_URI to:
# MLFLOW_TRACKING_URI=file:///absolute/path/to/Disease Prediction/mlruns
```

### Step 3: Download Data & Train Models

```bash
bash scripts/download_datasets.sh
MLFLOW_TRACKING_URI="file://$(pwd)/mlruns" python scripts/run_training.py --disease all
```

Training takes ~10–20 minutes per disease (Optuna, 80 trials, nested CV).

### Step 4: Start the API

```bash
MLFLOW_TRACKING_URI="file://$(pwd)/mlruns" \
  .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The startup log will show:
```
✓ Loaded heart model from models:/disease-prediction-heart/@latest
✓ Loaded diabetes model from models:/disease-prediction-diabetes/@latest
✓ SHAP explainer ready for heart (47 features)
```

### Step 5: Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Open: http://localhost:5173
```

### Step 6: Test Predictions

```bash
# Get a token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"admin"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Run a heart disease prediction
curl -X POST http://localhost:8000/api/v1/predict/heart \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65, "sex": 1, "cp": 3, "trestbps": 160, "chol": 310,
    "fbs": 1, "restecg": 1, "thalach": 120, "exang": 1,
    "oldpeak": 3.5, "slope": 2, "ca": 2, "thal": 2
  }'
```

---

## 20. Glossary

| Term | Definition |
|---|---|
| **AUC-PR** | Area Under the Precision-Recall Curve. Preferred metric for imbalanced datasets; not affected by the large true-negative count. |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve. Measures discrimination — probability that a randomly chosen positive patient ranks higher than a randomly chosen negative. |
| **Calibrated Probability** | A model output that has been post-processed via isotonic regression so that "70% probability" corresponds to an actual 70% disease prevalence in similar patients. |
| **SHAP** | SHapley Additive exPlanations — a game-theoretic method that assigns each feature a value representing its marginal contribution to a specific prediction. |
| **SMOTE** | Synthetic Minority Oversampling Technique — generates synthetic training examples in the minority class by interpolating between existing examples. |
| **MICE** | Multiple Imputation by Chained Equations — imputes missing values by training regression models iteratively on other features. |
| **eGFR** | Estimated Glomerular Filtration Rate — the primary measure of kidney function, in mL/min/1.73m². |
| **HOMA-IR** | Homeostatic Model Assessment of Insulin Resistance — a proxy for insulin resistance calculated from fasting glucose and insulin. |
| **Composite Risk Score** | A weighted combination of the model's calibrated probability (70%), risk velocity/trajectory (20%), and comorbidity burden (10%). |
| **Nested CV** | A cross-validation strategy with an outer loop for honest performance estimation and an inner loop for hyperparameter tuning, preventing optimistic bias. |
| **Wilson CI** | A confidence interval formula for proportions that remains reliable near 0 and 1, unlike the naive Wald interval. |
| **RBAC** | Role-Based Access Control — restricts system access based on a user's assigned role (admin / clinician / viewer). |
| **MLflow Model Registry** | A centralized store for tracking model versions, stages (Staging/Production), and metadata. |
| **Youden's J** | `Sensitivity + Specificity - 1` — a statistical measure for selecting optimal classification thresholds. |
