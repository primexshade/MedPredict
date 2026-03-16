# MedPredict API Reference

**Base URL:** `http://localhost:8000/api/v1`  
**OpenAPI docs:** `http://localhost:8000/docs`  
**Version:** 1.0.0  
**Auth:** Bearer JWT (`Authorization: Bearer <access_token>`)

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Prediction Endpoints](#2-prediction-endpoints)
3. [Analytics Endpoints](#3-analytics-endpoints)
4. [Patients Endpoints](#4-patients-endpoints)
5. [Reports Endpoints](#5-reports-endpoints)
6. [System Endpoints](#6-system-endpoints)
7. [Error Reference](#7-error-reference)
8. [Rate Limits](#8-rate-limits)
9. [Common Types](#9-common-types)

---

## 1. Authentication

### POST `/auth/login`

Authenticate a user and obtain a JWT token pair.

**Request body:**
```json
{
  "email": "admin@example.com",
  "password": "admin"
}
```

**Response `200 OK`:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Error `401 Unauthorized`:**
```json
{ "detail": "Invalid credentials" }
```

> **Token lifetimes:** Access token expires in 30 minutes. Refresh token expires in 7 days.  
> **Usage:** Include access token in all subsequent requests as `Authorization: Bearer <access_token>`.

---

### POST `/auth/logout`

Invalidate the current session (adds JTI to Redis blacklist).

**Auth required:** Yes  
**Response `200 OK`:**
```json
{ "message": "Logged out successfully" }
```

---

## 2. Prediction Endpoints

All prediction endpoints require authentication and are rate-limited to **20 requests/minute**.

### Common Response Schema (`PredictionResponse`)

All four disease endpoints return the same response structure:

```json
{
  "patient_id": "P-001",
  "disease": "heart",
  "risk_score": 0.63,
  "calibrated_probability": 0.71,
  "risk_category": "HIGH",
  "confidence_interval": [0.58, 0.82],
  "velocity": null,
  "top_features": [
    {
      "feature": "oldpeak",
      "value": 2.3,
      "shap_value": 0.18,
      "direction": "increases_risk",
      "rank": 1
    }
  ],
  "plain_english_summary": "Predicted heart risk: 71% (HIGH). Primary risk drivers: ST depression, age.",
  "clinical_action": {
    "action": "Specialist referral recommended",
    "timeframe": "1 month follow-up",
    "urgency": "high"
  },
  "model_version": "heart_v1.0",
  "cached": false
}
```

| Field | Type | Description |
|---|---|---|
| `patient_id` | `string \| null` | Echo of the submitted patient_id |
| `disease` | `string` | Disease key: `heart \| diabetes \| cancer \| kidney` |
| `risk_score` | `float [0,1]` | Composite score (model prob + velocity + comorbidity) |
| `calibrated_probability` | `float [0,1]` | Raw isotonic-calibrated model probability |
| `risk_category` | `string` | `LOW \| BORDERLINE \| MODERATE \| HIGH \| CRITICAL` |
| `confidence_interval` | `[float, float]` | 95% Wilson CI around calibrated probability |
| `velocity` | `float \| null` | Risk change from last visit (null if no history) |
| `top_features` | `FeatureContribution[]` | Top 5 SHAP feature contributions |
| `plain_english_summary` | `string` | Auto-generated natural language explanation |
| `clinical_action` | `object` | Recommended action, timeframe, urgency level |
| `model_version` | `string` | Model version string |
| `cached` | `boolean` | True if result was served from Redis cache |

---

### POST `/predict/heart`

Predict heart disease risk from ACC/AHA-aligned biomarkers.

**Auth required:** Yes  
**Rate limit:** 20/minute

**Request body:**
```json
{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1,
  "patient_id": "P-001"
}
```

**Field definitions:**

| Field | Type | Range | Description |
|---|---|---|---|
| `age` | float | 1–120 | Patient age in years |
| `sex` | int | 0–1 | 0 = Female, 1 = Male |
| `cp` | int | 0–3 | Chest pain type (0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic) |
| `trestbps` | float | 50–250 | Resting blood pressure (mmHg) |
| `chol` | float | 0–700 | Serum cholesterol (mg/dL) |
| `fbs` | int | 0–1 | Fasting blood sugar > 120 mg/dL (1=True) |
| `restecg` | int | 0–2 | Resting ECG (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy) |
| `thalach` | float | 40–250 | Maximum heart rate achieved |
| `exang` | int | 0–1 | Exercise-induced angina (1=Yes) |
| `oldpeak` | float | 0–10 | ST depression induced by exercise relative to rest |
| `slope` | int | 0–2 | Slope of peak exercise ST segment (0=Upsloping, 1=Flat, 2=Downsloping) |
| `ca` | int | 0–3 | Number of major vessels coloured by fluoroscopy |
| `thal` | int | 0–2 | Thalassemia (0=Normal, 1=Fixed defect, 2=Reversible defect) |
| `patient_id` | string? | — | Optional patient identifier (echoed in response, not stored) |

---

### POST `/predict/diabetes`

Predict Type 2 diabetes risk using PIMA Indian Diabetes Study features.

**Auth required:** Yes  
**Rate limit:** 20/minute

**Request body:**
```json
{
  "pregnancies": 6,
  "glucose": 148,
  "bloodpressure": 72,
  "skinthickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetespedigreefunction": 0.627,
  "age": 50,
  "patient_id": "P-002"
}
```

**Field definitions:**

| Field | Type | Range | Description |
|---|---|---|---|
| `pregnancies` | float | 0–20 | Number of times pregnant |
| `glucose` | float | 0–400 | Plasma glucose concentration (mg/dL, 2-hour OGTT) |
| `bloodpressure` | float | 0–200 | Diastolic blood pressure (mmHg) |
| `skinthickness` | float | 0–100 | Triceps skinfold thickness (mm). Use 0 if not measured. |
| `insulin` | float | 0–900 | 2-hour serum insulin (μU/mL). Use 0 if not measured. |
| `bmi` | float | 0–80 | Body Mass Index (weight in kg / height in m²) |
| `diabetespedigreefunction` | float | 0–3 | Diabetes pedigree function (genetic risk proxy) |
| `age` | float | 1–120 | Patient age in years |

> **Note on zeros:** In the PIMA dataset, physiological measurements (glucose, blood pressure, skin thickness, insulin, BMI) have zeros that encode missing values. The preprocessing pipeline automatically treats these as `NaN` and imputes them via MICE.

---

### POST `/predict/cancer`

Predict breast cancer malignancy (Malignant/Benign) using Wisconsin WDBC features.

**Auth required:** Yes  
**Rate limit:** 20/minute

**Request body (full 30-feature WDBC):**
```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "smoothness_mean": 0.1184,
  "compactness_mean": 0.2776,
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.07871,
  "radius_se": 1.095,
  "texture_se": 0.9053,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "smoothness_se": 0.006399,
  "compactness_se": 0.04904,
  "concavity_se": 0.05373,
  "concave_points_se": 0.01587,
  "symmetry_se": 0.03003,
  "fractal_dimension_se": 0.006193,
  "radius_worst": 25.38,
  "texture_worst": 17.33,
  "perimeter_worst": 184.6,
  "area_worst": 2019.0,
  "smoothness_worst": 0.1622,
  "compactness_worst": 0.6656,
  "concavity_worst": 0.7119,
  "concave_points_worst": 0.2654,
  "symmetry_worst": 0.4601,
  "fractal_dimension_worst": 0.1189,
  "patient_id": "P-003"
}
```

> **SE fields default to 0.0** if not available. For best accuracy, provide all 30 features.  
> **Cancer thresholds are more conservative** — the model is calibrated to err toward higher risk (lower CRITICAL/HIGH thresholds) to avoid missing malignancies.

---

### POST `/predict/kidney`

Predict chronic kidney disease (CKD) risk from clinical lab values.

**Auth required:** Yes  
**Rate limit:** 20/minute

**Request body:**
```json
{
  "age": 58.0,
  "bp": 80.0,
  "sg": 1.010,
  "al": 1.0,
  "su": 0.0,
  "bgr": 120.0,
  "bu": 36.0,
  "sc": 1.2,
  "sod": 137.0,
  "pot": 4.5,
  "hemo": 13.5,
  "pcv": 44.0,
  "wc": 7800.0,
  "rc": 5.2,
  "htn": 0,
  "dm": 0,
  "cad": 0,
  "appet": 1,
  "pe": 0,
  "ane": 0,
  "patient_id": "P-004"
}
```

**Field definitions:**

| Field | Type | Range | Description |
|---|---|---|---|
| `age` | float | 0–120 | Age in years |
| `bp` | float | 0–250 | Blood pressure (mmHg) |
| `sg` | float | 1.000–1.030 | Specific gravity of urine |
| `al` | float | 0–5 | Albumin (0–5 ordinal scale) |
| `su` | float | 0–5 | Sugar (0–5 ordinal scale) |
| `bgr` | float | 0–500 | Blood glucose random (mg/dL) |
| `bu` | float | 0–200 | Blood urea (mg/dL) |
| `sc` | float | 0–20 | Serum creatinine (mg/dL) |
| `sod` | float | 0–200 | Sodium (mEq/L). Default: 137 |
| `pot` | float | 0–15 | Potassium (mEq/L). Default: 4.5 |
| `hemo` | float | 0–20 | Haemoglobin (g/dL) |
| `pcv` | float | 0–60 | Packed cell volume (%). Default: 44 |
| `wc` | float | 0–30000 | White blood cell count (cells/cumm). Default: 7800 |
| `rc` | float | 0–10 | Red blood cell count (millions/cmm). Default: 5.2 |
| `htn` | int | 0–1 | Hypertension (1=Yes) |
| `dm` | int | 0–1 | Diabetes mellitus (1=Yes) |
| `cad` | int | 0–1 | Coronary artery disease (1=Yes) |
| `appet` | int | 0–1 | Appetite (0=Poor, 1=Good). Default: 1 |
| `pe` | int | 0–1 | Pedal oedema (1=Yes) |
| `ane` | int | 0–1 | Anaemia (1=Yes) |

---

## 3. Analytics Endpoints

**Auth required:** Yes  
**Rate limit:** 100/minute

### GET `/analytics/summary`

Returns population-level prediction statistics.

**Response `200 OK`:**
```json
{
  "total_predictions": 1247,
  "disease_breakdown": {
    "heart": 438,
    "diabetes": 312,
    "cancer": 284,
    "kidney": 213
  },
  "high_risk_count": 189
}
```

---

### GET `/analytics/clusters`

Returns PCA-projected patient phenotype clusters (Gaussian Mixture Model).

**Response `200 OK`:**
```json
{
  "clusters": [
    {
      "patient_id": "P-001",
      "x": 1.23,
      "y": -0.45,
      "cluster_id": 2,
      "risk_score": 0.72
    }
  ]
}
```

---

### GET `/analytics/comorbidity-rules`

Returns FP-Growth association rules for disease comorbidity patterns.

**Response `200 OK`:**
```json
{
  "rules": [
    {
      "antecedents": ["heart_disease", "diabetes"],
      "consequents": ["kidney_disease"],
      "support": 0.12,
      "confidence": 0.67,
      "lift": 3.2
    }
  ]
}
```

---

## 4. Patients Endpoints

**Auth required:** Yes  
**Rate limit:** 100/minute

### GET `/patients/`

List all patients in the registry.

**Response `200 OK`:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "mrn": "MRN-001234"
  }
]
```

### GET `/patients/{patient_id}`

Get a specific patient by UUID.

**Response `200 OK`:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "mrn": "MRN-001234"
}
```

**Error `404 Not Found`:**
```json
{ "detail": "Patient not found" }
```

---

## 5. Reports Endpoints

**Auth required:** Yes

### GET `/reports/`

Returns available report types for export (PDF generation).

---

## 6. System Endpoints

### GET `/health`

Health check — no auth required.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development"
}
```

---

## 7. Error Reference

All errors follow FastAPI's standard error format:

```json
{
  "detail": "Human-readable error message"
}
```

| HTTP Code | Meaning | Common Causes |
|---|---|---|
| `400 Bad Request` | Validation failure | Field out of range, missing required field |
| `401 Unauthorized` | Invalid/expired token | Missing/expired access token |
| `403 Forbidden` | Insufficient permissions | Role doesn't have required permission |
| `404 Not Found` | Resource not found | Invalid patient ID |
| `422 Unprocessable Entity` | Schema validation | Wrong type, constraint violation |
| `429 Too Many Requests` | Rate limit exceeded | > 20 prediction requests/minute |
| `503 Service Unavailable` | Model not loaded | Model not yet trained or failed to load |

---

## 8. Rate Limits

| Endpoint Group | Limit | Window |
|---|---|---|
| Authentication | 10/minute | Per IP |
| Prediction (`/predict/*`) | 20/minute | Per IP |
| Analytics, Patients, Reports | 100/minute | Per IP |
| Health check | Unlimited | — |

When a rate limit is exceeded, the API returns `429 Too Many Requests` with a `Retry-After` header indicating when the limit resets.

---

## 9. Common Types

### `FeatureContribution`

```typescript
interface FeatureContribution {
  feature: string;        // Clinical feature name (original or engineered)
  value: number;          // Patient's actual value for this feature
  shap_value: number;     // SHAP contribution (positive = increases risk)
  direction: "increases_risk" | "decreases_risk" | "neutral";
  rank: number;           // 1 = largest contributor
}
```

### `RiskCategory`

| Value | Heart Threshold | Diabetes | Cancer | Kidney |
|---|---|---|---|---|
| `LOW` | < 15% | < 20% | < 10% | < 15% |
| `BORDERLINE` | 15–30% | 20–35% | 10–25% | 15–30% |
| `MODERATE` | 30–55% | 35–55% | 25–50% | 30–55% |
| `HIGH` | 55–75% | 55–75% | 50–65% | 55–75% |
| `CRITICAL` | > 75% | > 75% | > 65% | > 75% |

### `ClinicalAction`

```typescript
interface ClinicalAction {
  action: string;     // Human-readable recommended action
  timeframe: string;  // Follow-up timeframe
  urgency: "low" | "medium" | "high" | "critical";
}
```

---

## Quick-Start: cURL Examples

```bash
# 1. Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"admin"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Heart disease prediction
curl -X POST http://localhost:8000/api/v1/predict/heart \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 2, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'

# 3. Diabetes prediction
curl -X POST http://localhost:8000/api/v1/predict/diabetes \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6, "glucose": 148, "bloodpressure": 72,
    "skinthickness": 35, "insulin": 0, "bmi": 33.6,
    "diabetespedigreefunction": 0.627, "age": 50
  }'
```
