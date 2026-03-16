# Model Card — Chronic Kidney Disease (CKD) Prediction

**Model name:** `disease-prediction-kidney`  
**Version:** 2 (trained on augmented dataset, March 2026)  
**Algorithm:** XGBoost with isotonic probability calibration  
**Task:** Binary classification (CKD / Not CKD)  
**Last trained:** 2026-03-17  
**Owner:** MedPredict ML Team

---

## Intended Use

### Primary Use Case

Predict whether a patient has chronic kidney disease (CKD) from 20 clinical and laboratory features, including a model-computed eGFR proxy using the simplified CKD-EPI formula. Intended for use in nephrology departments and primary care settings for early CKD detection.

### Intended Users

- Nephrologists and primary care physicians performing CKD screening
- Clinical staff operating the MedPredict platform (clinician role)

### Out-of-Scope Uses

- **Not** a CKD staging tool — does not classify CKD into Stages 1–5 (though `ckd_stage_proxy` is computed internally)
- **Not** validated for acute kidney injury (AKI) — built for chronic kidney disease
- **Not** for paediatric populations
- **Not** a replacement for specialist nephrology assessment

---

## Model Details

### Algorithm

XGBoost (`XGBClassifier`, with `scale_pos_weight` for class imbalance) wrapped in:
1. `ImbPipeline` (ColumnTransformer → XGBoost) — **no SMOTE** (dataset distribution allows direct `scale_pos_weight`)
2. `CalibratedClassifierCV(method="isotonic", cv=5)`

### Input Features

**Numeric features (14):**

| Feature | Unit | Range | Description |
|---|---|---|---|
| `age` | years | 2–90 | Patient age |
| `bp` | mmHg | 50–180 | Blood pressure |
| `sg` | — | 1.005–1.025 | Specific gravity of urine |
| `al` | ordinal | 0–5 | Albumin level (0–5 scale) |
| `su` | ordinal | 0–5 | Sugar level (0–5 scale) |
| `bgr` | mg/dL | 22–500 | Blood glucose (random) |
| `bu` | mg/dL | 1.5–391 | Blood urea |
| `sc` | mg/dL | 0.4–76 | Serum creatinine ← **strongest predictor** |
| `sod` | mEq/L | 111–163 | Sodium |
| `pot` | mEq/L | 2.5–47 | Potassium |
| `hemo` | g/dL | 3.1–17.8 | Haemoglobin |
| `pcv` | % | 9–54 | Packed cell volume |
| `wc` | cells/cumm | 2200–26400 | White blood cell count |
| `rc` | millions/cmm | 2.1–8 | Red blood cell count |

**Categorical features (6) — binary yes/no:**

| Feature | Description |
|---|---|
| `htn` | Hypertension |
| `dm` | Diabetes mellitus |
| `cad` | Coronary artery disease |
| `appet` | Appetite (0=Poor, 1=Good) |
| `pe` | Pedal oedema |
| `ane` | Anaemia |

### Engineered Features

| Feature | Formula | Clinical Basis |
|---|---|---|
| `egfr_proxy` | `186 × sc^(-1.154) × age^(-0.203)` | Simplified CKD-EPI eGFR formula — gold standard for kidney function staging (KDIGO 2012) |
| `ckd_stage_proxy` | `pd.cut(egfr_proxy, [0,15,30,45,60,90,∞]) → labels [5,4,3,2,1,0]` | Maps eGFR to CKD Stages 5–1 (5=worst, 0=normal) |
| `anemia_flag` | `hemoglobin < 12.0` | Anaemia is a direct consequence of EPO deficiency in advanced CKD |
| `hypertension_severe` | `bp > 90` | Diastolic BP > 90 accelerates CKD progression |

### Best Hyperparameters (Version 2)

```
n_estimators:       359
max_depth:          5
learning_rate:      0.0286
subsample:          0.601
colsample_bytree:   0.770
min_child_weight:   4
gamma:              0.261
reg_alpha:          6.863
reg_lambda:         0.016
scale_pos_weight:   (neg_count / pos_count)
```

---

## Training Data

| Attribute | Details |
|---|---|
| **Dataset** | UCI Chronic Kidney Disease (synthetically generated in UCI format) |
| **Original rows** | 400 rows |
| **After augmentation** | 849 rows |
| **Class balance (original)** | 62.5% CKD positive, 37.5% Not CKD |
| **SMOTE** | Disabled |
| **Train / Val / Test** | 70% / 15% / 15% (stratified) |

---

## Performance (Version 2 — Test Set, 128 rows)

| Metric | Value |
|---|---|
| **AUC-PR** | **1.0000** |
| **AUC-ROC** | **1.0000** |
| **Sensitivity** | **1.000** (100.0%) |
| **Specificity** | **1.000** (100.0%) |
| **F1 Score** | 1.000 |
| **MCC** | 1.000 |
| **Brier Score** | 0.000573 |
| **ECE** | 0.002234 |

### Context for Perfect Metrics

The perfect test set performance (AUC-PR = 1.0 across all 5 nested CV outer folds) reflects the nature of the underlying dataset:

- The UCI CKD dataset has very strong signal in creatinine (`sc`), haemoglobin (`hemo`), and the engineered `egfr_proxy` — biological associations that are extremely consistent
- The training dataset is synthetically generated using realistic clinical parameter distributions, which may produce cleanly separable class distributions

**Clinical implication:** This does not mean CKD is "trivially predictable" in real-world clinical practice — messy EMR data, non-standard lab reference ranges, and comorbidities make real-world performance lower. Independent validation on real hospital data is strongly recommended before clinical deployment.

---

## Risk Thresholds

| Category | Composite Score | Action |
|---|---|---|
| LOW | < 15% | Routine renal function monitoring annually |
| BORDERLINE | 15–30% | Lifestyle modification, dietary sodium/protein reduction |
| MODERATE | 30–55% | Nephrology referral for workup |
| HIGH | 55–75% | Specialist management, ACE inhibitor/ARB therapy evaluation |
| CRITICAL | > 75% | Urgent nephrology evaluation (possible dialysis planning) |

---

## Clinical Feature Importance Notes

The most clinically predictive features (per SHAP analysis) are typically:
1. **Serum creatinine (`sc`)** — direct measure of kidney filtration failure
2. **Haemoglobin (`hemo`)** + `anemia_flag` — anaemia of chronic disease
3. **`egfr_proxy`** — computed from creatinine and age (CKD-EPI)
4. **`ckd_stage_proxy`** — CKD staging via eGFR binning
5. **Blood urea (`bu`)** — elevated in renal failure (uraemia)

---

## Known Limitations

1. **Synthetic dataset**: The training data was synthetically generated rather than derived from a de-identified clinical database. Performance on genuine hospital EHR data should be independently validated.
2. **Perfect CV metrics**: AUC-PR = 1.0 across all folds suggests possible data leakage in the source dataset or overly clean synthetic data. Treat with caution until externally validated.
3. **Missing eGFR components**: The true CKD-EPI formula also uses sex and race as adjustment factors (race adjustment was officially deprecated in 2021). Our simplified formula uses only creatinine and age.
4. **`appet` encoding**: In the original UCI dataset, appetite is a string ("good"/"poor"). Our feature pipeline encodes this as 1/0. Missing appetite data defaults to 1=Good in `KidneyInput`.
