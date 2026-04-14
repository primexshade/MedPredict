# Model Card — Type 2 Diabetes Prediction

**Model name:** `disease-prediction-diabetes`  
**Version:** 2 (trained on augmented dataset, March 2026)  
**Algorithm:** LightGBM with isotonic probability calibration  
**Task:** Binary classification (diabetes / no diabetes)  
**Last trained:** 2026-03-17  
**Owner:** MedPredict ML Team

---

## Intended Use

### Primary Use Case

Predict the probability of Type 2 diabetes onset in adult patients using 8 metabolic markers from the PIMA Indian Diabetes Study protocol, augmented with clinically-derived engineered features based on ADA 2024 Standards of Care.

### Intended Users

- Endocrinologists and primary care physicians performing metabolic screening
- Diabetes prevention program coordinators

### Out-of-Scope Uses

- **Not** validated for Type 1 diabetes (autoimmune) — this model predicts T2DM only
- **Not** validated for populations outside the PIMA dataset demographic (women aged 21+, Pima Native American population) — broader validation is required before clinical deployment in other populations
- **Not** a gestational diabetes screening tool (different risk factors apply)

---

## Model Details

### Algorithm

LightGBM Gradient Boosted Trees (`LGBMClassifier`, `class_weight="balanced"`) wrapped in:
1. `ImbPipeline` (ColumnTransformer → SMOTE → LightGBM)
2. `CalibratedClassifierCV(method="isotonic", cv=5)`

LightGBM was selected over XGBoost for diabetes because:
- The all-numeric feature set benefits from LightGBM's leaf-wise tree growth
- LightGBM is faster to train on the smaller (~1,500 row) dataset
- Comparable accuracy to XGBoost with better training efficiency

### Input Features (8 clinical measurements)

| Feature | Type | Unit | Description |
|---|---|---|---|
| `pregnancies` | Continuous | Count | Number of times pregnant |
| `glucose` | Continuous | mg/dL | 2-hour plasma glucose (OGTT). **0 = missing in PIMA dataset** |
| `bloodpressure` | Continuous | mmHg | Diastolic blood pressure. **0 = missing** |
| `skinthickness` | Continuous | mm | Triceps skinfold thickness. **0 = missing** |
| `insulin` | Continuous | μU/mL | 2-hour serum insulin. **0 = missing** |
| `bmi` | Continuous | kg/m² | Body Mass Index. **0 = missing** |
| `diabetespedigreefunction` | Continuous | — | Genetic diabetes risk proxy (likelihood of diabetes based on family history) |
| `age` | Continuous | years | Age at examination |

> **Important note on zeros:** In the PIMA dataset, physiological zeros (glucose=0, BP=0, etc.) are data entry conventions for missing values, not true physiological readings. The data loader automatically converts these to `NaN`, and the IterativeImputer fills them using all other features.

### Engineered Features

| Feature | Formula | Clinical Basis |
|---|---|---|
| `obese` | `BMI ≥ 30` | WHO obesity classification (ADA 2024) |
| `severely_obese` | `BMI ≥ 35` | WHO Class II obesity |
| `glucose_prediabetic` | `glucose ∈ [100, 125]` | ADA 2024 IFG prediabetes range |
| `glucose_diabetic` | `glucose > 126` | ADA 2024 diabetes diagnostic threshold |
| `homa_ir_proxy` | `(glucose × insulin) / 405` | HOMA-IR insulin resistance (ADA formula, mg/dL) |
| `glucose_bmi` | `glucose × bmi` | Metabolic syndrome interaction |
| `preg_bmi` | `pregnancies × bmi` | Gestational metabolic burden proxy |
| `glucose_age_ratio` | `glucose / age` | Age-adjusted glucose intolerance |

### Best Hyperparameters (Version 2)

```
n_estimators:      553
max_depth:         4
learning_rate:     0.0050
num_leaves:        68
min_child_samples: 32
subsample:         0.844
colsample_bytree:  0.998
reg_alpha:         0.060
reg_lambda:        1.168
class_weight:      balanced
```

---

## Training Data

| Attribute | Details |
|---|---|
| **Dataset** | PIMA Indians Diabetes Database (Kaggle/UCI) |
| **Original rows** | 768 rows |
| **After augmentation** | 1,499 rows |
| **Class balance (original)** | 34.9% positive (diabetic), 65.1% negative |
| **SMOTE applied** | Yes — imbalance ratio ~1.87:1 (threshold: 2:1) |
| **Train / Val / Test** | 70% / 15% / 15% (stratified) |

---

## Performance (Version 2 — Test Set, 225 rows)

| Metric | Value |
|---|---|
| **AUC-PR** | **0.7790** |
| **AUC-ROC** | **0.8652** |
| **Sensitivity (Recall)** | **0.646** (64.6%) |
| **Specificity** | **0.849** (84.9%) |
| **PPV (Precision)** | 0.699 |
| **NPV** | 0.816 |
| **F1 Score** | 0.671 |
| **Brier Score** | 0.144 |
| **ECE** | 0.080 |

### Performance Context

The diabetes model has the lowest AUC-PR (0.779) of the four models. This is expected because:
1. The PIMA dataset is the smallest and most constrained demographic
2. Diabetes risk is genuinely difficult to predict from 8 biomarkers with high precision
3. In clinical practice, HbA1c (not available in PIMA) is the definitive diabetes screening test

The model still provides clinically useful risk stratification — particularly for identifying moderate-to-high risk patients who warrant HbA1c testing.

---

## Risk Thresholds

| Category | Composite Score Range | Action |
|---|---|---|
| LOW | < 20% | Annual screening |
| BORDERLINE | 20–35% | Lifestyle intervention, dietary counseling |
| MODERATE | 35–55% | HbA1c test recommended |
| HIGH | 55–75% | Endocrinologist referral |
| CRITICAL | > 75% | Immediate clinical evaluation |

---

## Known Limitations

1. **Dataset demographic**: The PIMA dataset consists entirely of adult women of Pima Native American heritage. Performance may differ significantly for other populations (different genetic background, lifestyle factors).
2. **Missing key markers**: HbA1c (gold standard for diabetes diagnosis) and OGTT test glucose (true 2-hour post-load) are not included in the feature set.
3. **Class imbalance**: The dataset is ~35% positive. While SMOTE addresses this during training, real-world diabetes prevalence varies widely (5–20% in different populations). Calibration may drift in populations with very different prevalence.
4. **Synthetic augmentation**: 731 synthetic rows added via Gaussian sampling. The all-missing-zero convention of the PIMA dataset adds complexity to augmentation fidelity.
