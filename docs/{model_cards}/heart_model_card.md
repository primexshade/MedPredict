# Model Card — Heart Disease Prediction

**Model name:** `disease-prediction-heart`  
**Version:** 2 (trained on augmented dataset, March 2026)  
**Algorithm:** XGBoost with isotonic probability calibration  
**Task:** Binary classification (disease present / not present)  
**Last trained:** 2026-03-17  
**Owner:** MedPredict ML Team

---

## Intended Use

### Primary Use Case

This model is intended as a **clinical decision support tool** to assist clinicians in identifying patients at elevated risk of significant coronary artery disease (CAD), based on 13 standardized ACC/AHA-aligned biomarkers collected during routine clinical examination.

### Intended Users

- Cardiologists and general practitioners performing cardiovascular risk screening
- Clinical support staff operating the MedPredict platform (with `clinician` role)

### Out-of-Scope Uses

- **Not** a diagnostic tool — predictions are probabilistic estimates, not diagnoses
- **Not** intended to replace clinical judgment or override cardiologist assessment
- **Not** validated for use in pediatric populations (< 18 years)
- **Not** validated for populations where the Cleveland/Kaggle dataset is unrepresentative (e.g., certain Asian, African, or South American populations)

---

## Model Details

### Algorithm

XGBoost Gradient Boosted Trees (`XGBClassifier`) wrapped in:
1. `ImbPipeline` (ColumnTransformer → SMOTE → XGBoost)
2. `CalibratedClassifierCV(method="isotonic", cv=5)` for probability calibration

### Input Features (13 clinical variables)

| Feature | Type | Range | Description |
|---|---|---|---|
| `age` | Continuous | 20–90 | Age in years |
| `sex` | Binary | 0/1 | 0 = Female, 1 = Male |
| `cp` | Ordinal | 0–3 | Chest pain type (0=Typical Angina → 3=Asymptomatic) |
| `trestbps` | Continuous | 80–220 | Resting blood pressure (mmHg) |
| `chol` | Continuous | 100–600 | Serum cholesterol (mg/dL) |
| `fbs` | Binary | 0/1 | Fasting blood sugar > 120 mg/dL |
| `restecg` | Ordinal | 0–2 | Resting ECG results |
| `thalach` | Continuous | 60–210 | Maximum heart rate achieved |
| `exang` | Binary | 0/1 | Exercise-induced angina |
| `oldpeak` | Continuous | 0–8 | ST depression induced by exercise |
| `slope` | Ordinal | 0–2 | Slope of peak exercise ST segment |
| `ca` | Ordinal | 0–4 | Major vessels coloured by fluoroscopy |
| `thal` | Ordinal | 0–2 | Thalassemia type |

### Engineered Features (added during training and inference)

| Feature | Formula | Clinical Basis |
|---|---|---|
| `age_chol_ratio` | `chol / age` | Age-cholesterol non-linear synergy (ACC/AHA 2019) |
| `hr_age_product` | `thalach × age` | Duke Treadmill Score proxy |
| `chol_high` | `chol > 240` | AHA borderline-high threshold |
| `chol_very_high` | `chol > 280` | AHA high-risk threshold |
| `tachycardia_flag` | `thalach > 100` | Resting tachycardia risk factor |
| `hypertension_flag` | `trestbps > 130` | AHA 2017 Stage 1 hypertension |
| `ischemia_severity` | `oldpeak × exang` | Combined ST + angina = ischemia indicator |

### Hyperparameters (Optuna Best Trial — Version 2)

```
n_estimators:     (Optuna-selected, typically 300–600)
max_depth:        3–8
learning_rate:    0.005–0.3 (log scale)
subsample:        0.6–1.0
colsample_bytree: 0.5–1.0
scale_pos_weight: neg_count / pos_count (handles class imbalance)
eval_metric:      aucpr
tree_method:      hist
```

---

## Training Data

| Attribute | Details |
|---|---|
| **Dataset** | Kaggle johnsmith88/heart-disease-dataset (Cleveland UCI + Kaggle combined) |
| **Original rows** | 1,025 rows |
| **After augmentation** | 1,999 rows (synthetic rows via per-class multivariate Gaussian sampling) |
| **Class balance** | Approximately 45% positive (disease), 55% negative |
| **Train / Val / Test split** | 70% / 15% / 15% (stratified) |
| **Augmentation method** | Per-class multivariate Gaussian sampling; domain constraints enforced |

---

## Performance (Version 2 — Test Set, 300 rows)

| Metric | Value |
|---|---|
| **AUC-PR** | **0.9439** |
| **AUC-ROC** | **0.9452** |
| **Sensitivity (Recall)** | **0.870** (87.0%) |
| **Specificity** | — |
| **Nested CV AUC-PR** | 0.9642 ± 0.0055 |
| Brier Score | (calibration quality) |
| F1 Score | — |

### Why AUC-PR as Primary Metric?

The dataset has mild class imbalance (~45%/55%). AUC-ROC can be misleadingly optimistic under imbalance. AUC-PR better captures performance on the positive (disease-present) class, which is the clinically important minority.

---

## Risk Thresholds

| Category | Composite Score Range | Recommended Action |
|---|---|---|
| LOW | < 15% | Routine screening in 12–24 months |
| BORDERLINE | 15–30% | Lifestyle modification counseling, 6-month follow-up |
| MODERATE | 30–55% | Clinical review + additional labs, 3-month follow-up |
| HIGH | 55–75% | Specialist referral, 1-month follow-up |
| CRITICAL | > 75% | Urgent clinical evaluation within 1 week |

---

## Known Limitations

1. **Population representation**: The Cleveland dataset is heavily skewed toward middle-aged North American males. The model may underperform for younger patients, women, or patients from different ethnic backgrounds.
2. **Synthetic augmentation**: 974 of the 1,999 training rows are Gaussian-sampled synthetic rows. While domain constraints are enforced, synthetic data may not perfectly replicate the joint distribution of rare clinical profiles.
3. **Missing features**: In clinical practice, some features (e.g., fluoroscopy vessel count `ca`, thalassemia `thal`) may not be routinely available. The MICE imputer handles missingness, but more missing features reduce accuracy.
4. **Static snapshot**: The model predicts from a single timepoint. Temporal risk trajectory is handled by the `RiskScorer` composite score (not the ML model itself).

---

## Ethical Considerations

- Clinicians should always explain prediction rationale using SHAP feature contributions
- Predictions should inform, not replace, clinical judgment
- Monitor for systematic differences in model performance across demographic groups
- Access to the prediction system is role-restricted (clinician minimum)
