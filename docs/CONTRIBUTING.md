# Contributing to MedPredict

Thank you for considering contributing to MedPredict. This document outlines the process and standards for contributing to this project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Testing Requirements](#testing-requirements)
6. [Submitting Changes](#submitting-changes)
7. [Adding a New Disease Model](#adding-a-new-disease-model)
8. [Clinical Validation Standards](#clinical-validation-standards)

---

## Code of Conduct

This project operates under a standard software engineering code of conduct. All contributors are expected to:
- Be respectful and professional in all communications
- Focus on the technical merit of contributions
- Adhere to medical data privacy principles (no real patient data in repos)
- Flag any clinical safety concerns immediately

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git
- Docker (optional, for full-stack local testing)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<org>/medpredict.git
cd medpredict

# 2. Create the Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# 3. Install Python dependencies (dev mode)
pip install -e ".[dev]"

# 4. Install frontend dependencies
cd frontend && npm install && cd ..

# 5. Copy environment config
cp .env.example .env
# Edit .env with your local values

# 6. Download datasets
bash scripts/download_datasets.sh

# 7. Train models locally (optional, fast mode)
python scripts/run_training.py --disease heart --trials 20
```

### Verify Your Setup

```bash
# Backend health check
pytest tests/unit/ -v

# Frontend type check
cd frontend && npx tsc --noEmit
```

---

## Development Workflow

We follow **trunk-based development** with short-lived feature branches.

```
main              ←── protected, CI required to merge
  └── feature/add-stroke-model
  └── fix/shap-calibrated-pipeline
  └── docs/api-reference-update
```

### Branch Naming Conventions

| Type | Pattern | Example |
|---|---|---|
| New feature | `feature/<description>` | `feature/add-stroke-model` |
| Bug fix | `fix/<description>` | `fix/shap-calibrated-pipeline` |
| Documentation | `docs/<description>` | `docs/api-reference` |
| Refactor | `refactor/<description>` | `refactor/pipeline-factory` |
| Hotfix | `hotfix/<description>` | `hotfix/critical-auth-bypass` |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(models): add stroke disease model with XGBoost pipeline
fix(scoring): clamp velocity contribution to prevent overflow
docs(api): document all prediction endpoint schemas
test(preprocessing): add kidney disease imputation unit tests
refactor(pipeline): extract SMOTE logic into standalone utility
```

---

## Code Standards

### Python

We use `ruff` for linting and `mypy` for type checking. Both must pass in CI.

```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/

# Auto-fix lints
ruff check src/ tests/ --fix
```

**Key rules enforced:**
- All functions must have type annotations
- No `print()` — use `logging.getLogger(__name__)`
- Maximum line length: 100 characters
- All public functions must have docstrings

**Clinical code standard:** Every engineered feature must include an inline comment citing the clinical guideline or literature source:

```python
# Good ✓
df["hypertension_flag"] = (df["trestbps"] > 130).astype(np.int8)
# AHA 2017: Stage 1 hypertension threshold for systolic BP

# Bad ✗
df["hyp"] = (df["trestbps"] > 130).astype(int)
```

### TypeScript / React

```bash
# Lint
cd frontend && npm run lint

# Type check
npx tsc --noEmit
```

- Use named exports (no default exports for components)
- All API response types must be defined in `frontend/src/services/api.ts`
- No `any` types — use proper interfaces from `api.ts`
- Use `React.FC<Props>` for all components

### SQL / Alembic

- All schema changes must be done via Alembic migrations (never ALTER TABLE manually)
- Migration descriptions must be human-readable: `alembic revision -m "add_comorbidity_index_to_predictions"`
- Never delete columns — mark as deprecated then remove in a later migration

---

## Testing Requirements

All PRs must maintain **≥ 80% test coverage** on new code.

### Test categories

```bash
# Fast unit tests (no I/O, no ML models)
pytest tests/unit/ -v

# Integration tests (requires PostgreSQL + Redis)
pytest tests/integration/ -v

# E2E tests (requires running backend + frontend)
pytest tests/e2e/ -v
```

### Writing Tests

**Unit tests** — test pure logic with synthetic DataFrames:

```python
def test_homa_ir_proxy_non_negative(sample_diabetes_df):
    result = apply_feature_engineering("diabetes", sample_diabetes_df)
    assert (result["homa_ir_proxy"] >= 0).all()
```

**Integration tests** — test full request lifecycle:

```python
async def test_predict_heart_returns_risk_category(client, auth_token):
    resp = await client.post(
        "/api/v1/predict/heart",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={...valid_payload...},
    )
    assert resp.status_code == 200
    assert resp.json()["risk_category"] in ["LOW", "BORDERLINE", "MODERATE", "HIGH", "CRITICAL"]
```

---

## Submitting Changes

### Pull Request Checklist

Before opening a PR, verify:

- [ ] `pytest tests/unit/ -v` passes (all green)
- [ ] `ruff check src/ tests/` has no errors
- [ ] `mypy src/` has no errors
- [ ] `cd frontend && npx tsc --noEmit` has no errors
- [ ] New functions have docstrings
- [ ] New clinical features cite their source guideline
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] If adding a new model: model card created in `docs/{model_cards}/`

### PR Description Template

```markdown
## Summary
Brief description of what changed and why.

## Clinical Impact
- [ ] No clinical behaviour changed (pure refactor/docs)
- [ ] Changed risk thresholds (requires clinical review)
- [ ] Added new disease model (requires model card)
- [ ] Changed feature engineering (requires clinical citation)

## Testing
- Unit tests added/updated: Yes/No
- Integration tests: Yes/No
- Manual testing steps: ...

## Screenshots
(For frontend changes)
```

---

## Adding a New Disease Model

To add a new disease (e.g., stroke), follow these steps in order:

### 1. Data

- Place raw CSV in `data/raw/<disease>.csv`
- Implement `load_<disease>()` in `src/data/load.py`
- Add domain constraints (`<DISEASE>_CONSTRAINTS`) and a `_validate_constraints()` call

### 2. Feature Engineering

- Add `engineer_<disease>_features(df)` in `src/features/engineering.py`
- Register it in `FEATURE_ENGINEERS` dict
- Each feature must cite a clinical guideline

### 3. Preprocessing Pipeline

- Add an entry to `FEATURE_CONFIG` in `src/features/pipeline.py`:
  ```python
  "stroke": {
      "numeric": [...],
      "categorical": [...],
      "target": "target",
      "use_smote": True,
      "imbalance_ratio_threshold": 2,
  }
  ```

### 4. Training

- Add `"stroke": lambda trial, spw: XGBClassifier(...)` to `MODEL_BUILDERS` in `src/models/train.py`
- Add the loader to `LOADERS` in `scripts/run_training.py`
- Train and verify AUC-PR ≥ 0.75

### 5. Risk Scoring

- Add thresholds to `RISK_THRESHOLDS` in `src/scoring/risk_scorer.py`
- Calibrate thresholds to maximize Youden's J on validation cohort

### 6. API

- Add `StrokeInput(BaseModel)` schema in `src/api/routers/predict.py`
- Add `@router.post("/stroke")` endpoint
- Register in `settings.registered_models`

### 7. Frontend

- Add a new tab to `frontend/src/pages/Predict.tsx`
- Add field definitions to `DISEASE_FORMS` constant
- Add API call to `predictAPI` in `frontend/src/services/api.ts`

### 8. Documentation

- Create `docs/{model_cards}/stroke_model_card.md` (see [Model Card Template](#))
- Update `docs/API_REFERENCE.md` with the new endpoint
- Update `README.md` diseases section

### 9. Testing

- Add unit tests for feature engineering
- Add integration tests for the prediction endpoint
- Document model performance in the model card

---

## Clinical Validation Standards

Any contribution that changes model outputs, thresholds, or feature engineering must:

1. **Not degrade AUC-PR** below the baseline recorded in the model card
2. **Cite clinical sources** for any new or modified features
3. **Maintain calibration quality** — Brier score must not worsen by > 0.01
4. **Preserve sensitivity** for cancer model — sensitivity must remain ≥ 90%

These standards exist because MedPredict is designed for clinical decision support. Degrading model performance has real patient safety implications.
