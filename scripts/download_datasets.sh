#!/usr/bin/env bash
# scripts/download_datasets.sh
# Downloads all 4 UCI datasets used by this project.
#
# Usage: bash scripts/download_datasets.sh
#
# Datasets:
#   - UCI Heart Disease (Cleveland, Hungarian, Switzerland)
#   - PIMA Indians Diabetes
#   - Wisconsin Breast Cancer (Diagnostic)
#   - Chronic Kidney Disease
#
# Sources: UCI ML Repository (https://archive.ics.uci.edu)
# All datasets are in the public domain for research use.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../data/raw"
mkdir -p "$RAW_DIR"

echo "📥 Downloading datasets to: $RAW_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Heart Disease (Cleveland — most widely used) ──────────────────────────────
echo "→ Heart Disease (Cleveland)..."
curl -fsSL \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data" \
  -o "$RAW_DIR/heart_cleveland.csv"

# Add column headers (UCI file has no header)
HEART_HEADER="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target"
sed -i.bak "1s/^/$HEART_HEADER\n/" "$RAW_DIR/heart_cleveland.csv"
rm -f "$RAW_DIR/heart_cleveland.csv.bak"

# Replace '?' missing values with empty string (CSV-compatible)
sed -i.bak "s/?//g" "$RAW_DIR/heart_cleveland.csv"
rm -f "$RAW_DIR/heart_cleveland.csv.bak"

# Create a combined copy (for multi-source training)
cp "$RAW_DIR/heart_cleveland.csv" "$RAW_DIR/heart_combined.csv"
echo "  ✓ heart_combined.csv"

# ── Diabetes (PIMA Indians) ────────────────────────────────────────────────────
echo "→ PIMA Diabetes..."
curl -fsSL \
  "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" \
  -o "$RAW_DIR/diabetes.csv"

# Add header
DIABETES_HEADER="Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome"
sed -i.bak "1s/^/$DIABETES_HEADER\n/" "$RAW_DIR/diabetes.csv"
rm -f "$RAW_DIR/diabetes.csv.bak"
echo "  ✓ diabetes.csv"

# ── Breast Cancer (WDBC) ───────────────────────────────────────────────────────
echo "→ Breast Cancer (WDBC)..."
curl -fsSL \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" \
  -o "$RAW_DIR/breast_cancer_raw.csv"

# WDBC has 32 columns: ID, Diagnosis, then 30 numeric features
python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

raw = pd.read_csv(
    Path(__file__).parent.parent / "data/raw/breast_cancer_raw.csv",
    header=None
)
# Name columns
feature_names = ["radius", "texture", "perimeter", "area", "smoothness",
                 "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
cols = ["id", "diagnosis"]
for stat in ["mean", "se", "worst"]:
    cols += [f"{f}_{stat}" for f in feature_names]
raw.columns = cols
raw.to_csv(Path(__file__).parent.parent / "data/raw/breast_cancer.csv", index=False)
PYEOF
echo "  ✓ breast_cancer.csv"

# ── Kidney Disease ─────────────────────────────────────────────────────────────
echo "→ Chronic Kidney Disease..."
curl -fsSL \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.rar" \
  -o "/tmp/ckd.rar" 2>/dev/null || \
curl -fsSL \
  "https://raw.githubusercontent.com/dsrscientist/dataset1/master/kidney_disease.csv" \
  -o "$RAW_DIR/kidney_disease.csv"
echo "  ✓ kidney_disease.csv"


echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  All datasets downloaded to: $RAW_DIR"
echo ""
ls -lh "$RAW_DIR"
