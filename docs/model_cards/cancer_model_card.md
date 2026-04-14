# Model Card — Breast Cancer (Malignant/Benign) Prediction

**Model name:** `disease-prediction-cancer`  
**Version:** 2 (trained on augmented dataset, March 2026)  
**Algorithm:** XGBoost with isotonic probability calibration  
**Task:** Binary classification (Malignant / Benign)  
**Last trained:** 2026-03-17  
**Owner:** MedPredict ML Team

---

## Intended Use

### Primary Use Case

Predict whether a breast mass is malignant or benign based on fine needle aspirate (FNA) biopsy measurements — the 30 Wisconsin Diagnostic Breast Cancer (WDBC) features covering 10 cell nucleus properties in three statistical moments (mean, SE, worst case).

### Intended Users

- Oncologists and radiologists reviewing FNA biopsy results
- Clinical staff operating the MedPredict platform (clinician role)

### Out-of-Scope Uses

- **Not** a screening tool for patients without a detectable breast mass — WDBC features require an FNA biopsy procedure first
- **Not** a substitute for biopsy pathology report — this model augments, not replaces, pathological examination
- **Not** validated for other cancer types (only Wisconsin breast cancer data was used)
- **Not** for use without clinical context — a high-risk score on a clearly benign cyst warrants additional clinical review

---

## Model Details

### Algorithm

XGBoost (`XGBClassifier`) wrapped in:
1. `ImbPipeline` (ColumnTransformer → XGBoost) — **no SMOTE** (dataset is relatively balanced at 63%/37%)
2. `CalibratedClassifierCV(method="isotonic", cv=5)`

### Input Features (30 WDBC numeric features)

The 30 features are organized as 3 statistical moments × 10 morphological properties of cell nuclei computed from FNA images:

**10 Morphological Properties:**

| Property | Description |
|---|---|
| `radius` | Mean of distances from center to points on the perimeter |
| `texture` | Standard deviation of gray-scale values |
| `perimeter` | Perimeter of the nucleus |
| `area` | Area of the nucleus |
| `smoothness` | Local variation in radius lengths |
| `compactness` | `perimeter² / area - 1.0` |
| `concavity` | Severity of concave portions of the contour |
| `concave_points` | Number of concave portions of the contour |
| `symmetry` | Symmetry of the nucleus |
| `fractal_dimension` | "Coastline approximation" - 1 |

**3 Statistical Moments (per property):**
- `_mean` — Mean across all cells in the image
- `_se` — Standard error
- `_worst` — Worst (largest) value among all cells

Total: 10 × 3 = **30 features**

### Engineered Features

| Feature | Formula | Clinical Basis |
|---|---|---|
| `radius_deterioration` | `radius_worst / radius_mean` | Relative cellular enlargement in worst-case cells |
| `perimeter_deterioration` | `perimeter_worst / perimeter_mean` | Nuclear border irregularity progression |
| `area_deterioration` | `area_worst / area_mean` | Tumour cell size heterogeneity |
| `concavity_deterioration` | `concavity_worst / concavity_mean` | Structural malignancy indicator |
| `concave_points_deterioration` | `concave_points_worst / concave_points_mean` | Complexity of nuclear contour |
| `shape_irregularity` | `perimeter_mean² / (4π × area_mean)` | Isoperimetric quotient — circular = 1.0, irregular > 1.0 (Wolberg 1995) |

---

## Training Data

| Attribute | Details |
|---|---|
| **Dataset** | Kaggle erdemtaha/cancer-data (Wisconsin WDBC format) |
| **Original rows** | 569 rows |
| **After augmentation** | 1,100 rows |
| **Class balance (original)** | 37.3% Malignant (M), 62.7% Benign (B) |
| **SMOTE** | Disabled (ratio is 1.68:1, below threshold) |
| **Train / Val / Test** | 70% / 15% / 15% (stratified) |

---

## Performance (Version 2 — Test Set, 165 rows)

| Metric | Value |
|---|---|
| **AUC-PR** | **0.9931** |
| **AUC-ROC** | **0.9962** |
| **Sensitivity (Recall for Malignant)** | **0.967** (96.7%) |
| Nested CV AUC-PR (Fold 1) | 0.9996 |
| Nested CV AUC-PR (Fold 2) | 0.9978 |

### Why Sensitivity Matters Most for Cancer

The cost of a **false negative** (predicting Benign when actually Malignant) vastly exceeds the cost of a **false positive** (predicting Malignant when Benign — triggers additional clinical evaluation). Accordingly:
- Cancer risk thresholds are **more conservative** (lower cutoffs) than other diseases
- The HIGH threshold is 50–65% (vs 55–75% for other diseases)
- The CRITICAL threshold is >65% (vs >75%) 

A sensitivity of 96.7% means the model correctly identifies ~97% of malignant cases, missing only ~3%.

---

## Risk Thresholds (More Conservative Than Other Diseases)

| Category | Composite Score Range | Action |
|---|---|---|
| LOW | < 10% | Routine follow-up |
| BORDERLINE | 10–25% | Repeat imaging in 6 months |
| MODERATE | 25–50% | Additional diagnostic workup |
| HIGH | 50–65% | Biopsy / specialist referral |
| CRITICAL | > 65% | Urgent oncology referral within 1 week |

---

## Known Limitations

1. **FNA dependency**: This model cannot be used without FNA biopsy measurements — it does not operate on imaging alone
2. **Limited to WDBC features**: Does not incorporate patient age, family history, BRCA status, hormone receptor status, or lymph node involvement — all of which are clinically relevant
3. **Synthetic augmentation**: 531 synthetic rows added. FNA measurements are highly correlated (mean/SE/worst of same property); the multivariate Gaussian sampler captures this but may not perfectly replicate rare morphological profiles
4. **Perfect-appearing nested CV**: Outer CV AUC-PR of ~0.999 may suggest the dataset separation is clean; real-world performance on more heterogeneous populations should be validated

---

## Ethical Considerations

- **Never** use this model as a standalone diagnostic — it is a decision support tool only
- A BORDERLINE result should trigger clinical follow-up, not automatic discharge
- Malignancy predictions must be communicated to patients with full clinical context
