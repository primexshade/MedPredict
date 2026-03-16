# Deployment Guide

This guide covers deploying MedPredict in local development, staging, and production environments.

---

## Table of Contents

1. [Local Development](#1-local-development)
2. [Docker Compose (Staging)](#2-docker-compose-staging)
3. [Production Deployment](#3-production-deployment)
4. [Environment Variables Reference](#4-environment-variables-reference)
5. [Database Migrations](#5-database-migrations)
6. [Model Training in Deployment](#6-model-training-in-deployment)
7. [Scaling Considerations](#7-scaling-considerations)
8. [Health Checks](#8-health-checks)
9. [Rollback Procedures](#9-rollback-procedures)

---

## 1. Local Development

### Prerequisites

| Tool | Minimum Version | Install |
|---|---|---|
| Python | 3.11 | `pyenv install 3.11` |
| Node.js | 18 LTS | `nvm install 18` |
| Docker | 20.10 | [docker.com](https://www.docker.com) |
| Git | 2.38 | `brew install git` |

### Setup

```bash
# Clone repo
git clone https://github.com/<org>/medpredict.git
cd medpredict

# Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Environment config
cp .env.example .env
# Set: MLFLOW_TRACKING_URI=file:///path/to/project/mlruns

# Download training data
bash scripts/download_datasets.sh

# Train models (reduced trials for speed)
python scripts/run_training.py --disease all --trials 20

# Start backend
uvicorn src.api.main:app --reload --port 8000

# In a new terminal — Start frontend
cd frontend && npm install && npm run dev
```

**Application URLs:**
- Frontend: http://localhost:5173
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Login: `admin@example.com` / `admin`

> **Note:** Redis and PostgreSQL are optional in local development. The API gracefully falls back to in-memory stubs when they are unavailable.

---

## 2. Docker Compose (Staging)

Docker Compose runs the complete stack: API, frontend, PostgreSQL, Redis, MLflow server, and Nginx reverse proxy.

### Quick Start

```bash
# Build and start all services
docker compose -f docker/docker-compose.yml up -d

# Check status
docker compose -f docker/docker-compose.yml ps

# View logs
docker compose -f docker/docker-compose.yml logs -f api

# Stop all services
docker compose -f docker/docker-compose.yml down
```

### Services

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI backend (Uvicorn) |
| `frontend` | 3000 | React app served by Nginx |
| `postgres` | 5432 | PostgreSQL 15 |
| `redis` | 6379 | Redis 7 (prediction cache) |
| `mlflow` | 5001 | MLflow tracking server |
| `nginx` | 80, 443 | Reverse proxy (HTTPS termination) |

### Environment for Docker Compose

Create `docker/.env.docker` (never commit this file):

```bash
DATABASE_URL=postgresql://medpredict:securepassword@postgres:5432/medpredict
REDIS_URL=redis://redis:6379/0
MLFLOW_TRACKING_URI=http://mlflow:5000
JWT_SECRET=change-this-to-a-32-char-random-string!!
ENVIRONMENT=staging
```

### Running Migrations in Docker

```bash
docker compose -f docker/docker-compose.yml run --rm api alembic upgrade head
```

---

## 3. Production Deployment

### Infrastructure Requirements

| Component | Minimum Spec | Recommended |
|---|---|---|
| API server | 2 vCPU, 4 GB RAM | 4 vCPU, 8 GB RAM |
| Database | PostgreSQL 15 (managed) | AWS RDS / GCP Cloud SQL |
| Cache | Redis 7 (managed) | AWS ElastiCache / GCP Memorystore |
| Model storage | 500 MB SSD | GCS bucket / S3 |
| Frontend CDN | — | CloudFront / Cloud CDN |

### Docker Production Build

```bash
# Build production image
docker build -f docker/Dockerfile -t medpredict-api:latest .

# Tag and push to registry
docker tag medpredict-api:latest ghcr.io/<org>/medpredict-api:v1.0.0
docker push ghcr.io/<org>/medpredict-api:v1.0.0
```

### Production Environment Variables

Set these via your cloud provider's secret manager (never in code):

```bash
ENVIRONMENT=production
DATABASE_URL=postgresql://<user>:<pass>@<host>:5432/medpredict
REDIS_URL=redis://<host>:6379/0
MLFLOW_TRACKING_URI=file:///app/mlruns    # or http://mlflow-server:5000
JWT_SECRET=<64-char random string>
LOG_LEVEL=WARNING
CORS_ORIGINS=["https://medpredict.yourdomain.com"]
GCP_PROJECT_ID=your-project-id           # if using GCP features
GCS_BUCKET_NAME=medpredict-models        # if using GCS model storage
```

### Kubernetes Deployment (Example)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medpredict-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: medpredict-api
  template:
    spec:
      containers:
      - name: api
        image: ghcr.io/<org>/medpredict-api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: medpredict-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30   # Wait for model loading
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

> **Important:** The readiness probe `initialDelaySeconds` should be set to at least 30 seconds to allow `preload_models()` to complete. If SHAP setup is loading large datasets, increase to 60 seconds.

---

## 4. Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ENVIRONMENT` | No | `development` | `development \| staging \| production` |
| `LOG_LEVEL` | No | `INFO` | `DEBUG \| INFO \| WARNING \| ERROR` |
| `DATABASE_URL` | Yes (prod) | `postgresql://user:password@localhost:5432/disease_prediction` | PostgreSQL connection string |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string |
| `MLFLOW_TRACKING_URI` | No | `file:///path/to/mlruns` | MLflow tracking URI |
| `JWT_SECRET` | **Yes** | `change-me-...` | JWT signing secret (min 32 chars) |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `30` | Access token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | `7` | Refresh token lifetime |
| `CORS_ORIGINS` | No | `["http://localhost:5173"]` | Allowed CORS origins (JSON array) |
| `PREDICTION_CACHE_TTL_SECONDS` | No | `300` | Redis cache TTL |
| `RATE_LIMIT_GENERAL` | No | `100/minute` | General endpoint rate limit |
| `RATE_LIMIT_PREDICTION` | No | `20/minute` | Prediction endpoint rate limit |
| `SMTP_HOST` | No | `smtp.gmail.com` | Email server (for notifications) |
| `SMTP_PORT` | No | `587` | Email server port |
| `SMTP_USER` | No | `""` | Email username |
| `SMTP_PASSWORD` | No | `""` | Email password |
| `GCP_PROJECT_ID` | No | `""` | GCP project (production only) |
| `GCS_BUCKET_NAME` | No | `""` | GCS bucket for model storage |

---

## 5. Database Migrations

Migrations are managed via Alembic. Always run migrations before starting the API.

```bash
# Apply all pending migrations
alembic upgrade head

# Create a new migration (after modifying src/db/models.py)
alembic revision --autogenerate -m "add_comorbidity_to_predictions"

# Check current migration state
alembic current

# View migration history
alembic history

# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade abc123def456
```

> **Production rule:** Never run `alembic downgrade` in production without a tested rollback plan.

---

## 6. Model Training in Deployment

Models are trained offline and loaded into the API at startup from the MLflow registry.

### Training

```bash
# Full training (recommended: 80 trials)
python scripts/run_training.py --disease all --trials 80

# Fast training (development, 20 trials)
python scripts/run_training.py --disease all --trials 20

# Single disease
python scripts/run_training.py --disease heart --trials 80
```

### Model Loading

The API loads models from `MLFLOW_TRACKING_URI` at startup. It tries these aliases in order:
```
models:/disease-prediction-{disease}@latest
models:/disease-prediction-{disease}/1
models:/disease-prediction-{disease}/2
models:/disease-prediction-{disease}/3
```

If no model is found, the endpoint returns `503 Service Unavailable` with detail: `"Model for '{disease}' is not loaded. Run training first."`

### Data Augmentation (Optional)

```bash
# Augment all datasets before training (recommended for small datasets)
python scripts/augment_data.py --disease all

# Or with a custom multiplication factor
python scripts/augment_data.py --n-factor 2.0
```

---

## 7. Scaling Considerations

### API Horizontal Scaling

The API is **stateless** (JWT auth, Redis cache, MLflow file store). You can run multiple replicas:

```bash
# Docker Compose scaling
docker compose -f docker/docker-compose.yml up -d --scale api=3
```

**Shared state requirements:**
- Redis must be a shared instance (not per-replica)
- MLflow `mlruns/` directory must be on shared storage (NFS, EFS, GCS FUSE) or a central MLflow server

### SHAP Memory Usage

Each SHAP TreeExplainer holds a 50-row background dataset and the tree model in memory. With 4 disease models:
- Estimated memory per API instance: ~500 MB—1.5 GB
- Plan accordingly for container limits

### Redis Cache

Predictions are cached for 5 minutes by content hash. In high-throughput scenarios, ensure Redis has sufficient memory:

```
# Estimate: ~2 KB per cached prediction
# 10,000 concurrent unique predictions → ~20 MB Redis memory required
```

---

## 8. Health Checks

### API Health Check

```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","version":"1.0.0","environment":"development"}
```

### Model Loading Verification

```bash
# Attempt a prediction — if model loaded, returns result; if not, returns 503
curl -X POST http://localhost:8000/api/v1/predict/heart \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"age":50,"sex":1,"cp":0,"trestbps":120,"chol":200,"fbs":0,"restecg":0,"thalach":150,"exang":0,"oldpeak":0,"slope":1,"ca":0,"thal":1}'
```

### Database Connectivity

```bash
# From inside the container
python -c "from src.db.session import engine; engine.connect(); print('DB OK')"
```

---

## 9. Rollback Procedures

### API Rollback

```bash
# Roll back to previous Docker image
docker compose -f docker/docker-compose.yml stop api
docker tag medpredict-api:v0.9.0 medpredict-api:latest
docker compose -f docker/docker-compose.yml up -d api
```

### Database Rollback

```bash
# Always create a backup before migrating in production
pg_dump -h $DB_HOST -U $DB_USER medpredict > backup_$(date +%Y%m%d%H%M%S).sql

# Rollback Alembic one step
alembic downgrade -1
```

### Model Rollback

```bash
# MLflow model versions are immutable — simply load a previous version
# In src/api/routers/predict.py, the alias cascade tries /1, /2, /3
# To force a specific version, update MLFLOW_TRACKING_URI or model alias
```
