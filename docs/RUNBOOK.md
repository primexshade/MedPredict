# Operational Runbook — MedPredict

**Audience:** On-call engineers and DevOps  
**Last Updated:** March 2026

This runbook covers how to diagnose and resolve the most common operational issues with MedPredict in production.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Monitoring & Alerting](#2-monitoring--alerting)
3. [Common Issues](#3-common-issues)
4. [Deployment Procedures](#4-deployment-procedures)
5. [Backup & Recovery](#5-backup--recovery)
6. [Escalation Matrix](#6-escalation-matrix)

---

## 1. System Overview

```
Traffic → Nginx (443) → FastAPI (8000) → PostgreSQL (5432)
                                       → Redis (6379)
                                       → MLflow file store / server
```

**Key processes:**
- `uvicorn` — ASGI server for FastAPI (PID tracked in process manager)
- `nginx` — Reverse proxy and TLS termination
- `postgres` — Primary database
- `redis-server` — Prediction cache

**Service URLs:**
- Frontend: `https://medpredict.yourdomain.com`
- API: `https://medpredict.yourdomain.com/api/v1`
- Health check: `https://medpredict.yourdomain.com/health`
- API Docs: `https://medpredict.yourdomain.com/docs`

---

## 2. Monitoring & Alerting

### Key Metrics to Monitor

| Metric | Warning Threshold | Critical Threshold |
|---|---|---|
| API response time (p95) | > 2s | > 5s |
| Prediction endpoint response time | > 3s | > 8s |
| Error rate (5xx) | > 1% | > 5% |
| Redis hit rate | < 50% | < 20% |
| PostgreSQL connection pool utilization | > 70% | > 90% |
| Container memory (API) | > 3 GB | > 4 GB |
| Container CPU (API) | > 70% | > 90% |

### Health Check Endpoints

```bash
# API health
curl https://medpredict.yourdomain.com/health

# Database connectivity (from container)
pg_isready -h $DB_HOST -p 5432

# Redis connectivity
redis-cli -h $REDIS_HOST ping
```

---

## 3. Common Issues

---

### 🔴 Issue: `503 Service Unavailable` — "Model is not loaded"

**Symptom:** Prediction requests return HTTP 503 with `"Model for 'heart' is not loaded. Run training first."`

**Cause:** The MLflow model registry has no model for the requested disease, or the model file path is unreachable.

**Diagnosis:**
```bash
# Check MLflow runs directory exists and has model artifacts
ls -la mlruns/models/

# Check which models are registered
.venv/bin/python -c "
import mlflow
mlflow.set_tracking_uri('file:///path/to/mlruns')
client = mlflow.tracking.MlflowClient()
for m in client.search_registered_models():
    print(m.name, [v.version for v in m.latest_versions])
"

# Check API logs for loading errors
docker compose logs api | grep -E "(ERROR|WARNING|✗|✓)"
```

**Resolution:**
```bash
# Train the missing model(s)
python scripts/run_training.py --disease heart --trials 50

# Restart API to reload models
docker compose restart api
```

---

### 🔴 Issue: API Fails to Start — `OSError: [Errno 48] Address already in use`

**Symptom:** API container fails to start with port conflict error.

**Diagnosis:**
```bash
lsof -ti :8000
```

**Resolution:**
```bash
# Kill the process using port 8000
kill -9 $(lsof -ti :8000)

# Restart API
docker compose restart api
```

---

### 🟡 Issue: Prediction Response Time > 3s

**Symptom:** `/predict/*` endpoints are consistently slow.

**Diagnosis:**
```bash
# Check if Redis cache is working (if cache hits are low, every request runs inference)
redis-cli info stats | grep keyspace_hits
redis-cli info stats | grep keyspace_misses

# Check API thread pool usage (SHAP + model inference runs in thread pool)
# Look for "ThreadPoolExecutor" backlog in logs
docker compose logs api | grep -i "thread"

# Profile memory pressure
docker stats medpredict_api
```

**Resolution options:**
- If Redis is down: restore Redis (cache miss → every request is slow)
- If thread pool is saturated: increase `max_workers` in `predict.py` `ThreadPoolExecutor`
- If memory is near limit: increase container memory limit and scale horizontally

---

### 🟡 Issue: `422 Unprocessable Entity` on Prediction

**Symptom:** Prediction endpoint returns 422 with validation error detail.

**Diagnosis:**
```bash
# The response body will contain the exact field that failed validation
# Example:
{
  "detail": [
    {
      "loc": ["body", "chol"],
      "msg": "ensure this value is less than or equal to 700",
      "type": "value_error.number.not_le"
    }
  ]
}
```

**Resolution:** This is a client-side issue (invalid input). No server action needed. Check the [API Reference](./API_REFERENCE.md) for valid field ranges. Common issues:
- Cholesterol value > 700 (very rare but possible in severe hypercholesterolaemia — contact engineering to adjust constraint if clinically justified)
- Cancer feature out of non-negative range

---

### 🟡 Issue: Redis Connection Refused / Cache Not Working

**Symptom:** Logs show `Redis connection refused` but predictions still work (graceful fallback active).

**Diagnosis:**
```bash
# Check Redis status
redis-cli -h $REDIS_HOST -p 6379 ping

# Check Redis logs
docker compose logs redis | tail -30
```

**Resolution:**
```bash
# Restart Redis
docker compose restart redis

# Verify connectivity from API container
docker compose exec api redis-cli -h redis ping
```

> **Note:** The API gracefully falls back to a no-op stub when Redis is unavailable. Predictions still work — just without caching (higher latency on repeated requests).

---

### 🟡 Issue: PostgreSQL `too many connections`

**Symptom:** API logs show `FATAL: remaining connection slots are reserved for non-replication superuser connections`.

**Diagnosis:**
```bash
# Check active connections
psql $DATABASE_URL -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

# Check max connections
psql $DATABASE_URL -c "SHOW max_connections;"
```

**Resolution:**
```bash
# 1. Immediately: terminate idle connections
psql $DATABASE_URL -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < NOW() - INTERVAL '5 minutes';"

# 2. Long-term: add PgBouncer connection pooler in front of PostgreSQL
# 3. Or: increase max_connections in postgresql.conf (requires restart)
```

---

### 🔴 Issue: Database Migration Failed — API and DB Schema Out of Sync

**Symptom:** API returns `500 Internal Server Error` with `ProgrammingError: column "xyz" does not exist`.

**Diagnosis:**
```bash
# Check current Alembic version
alembic current

# Check pending migrations
alembic history | head -20
```

**Resolution:**
```bash
# Run pending migrations (always do this in a maintenance window)
alembic upgrade head

# If a migration is broken, rollback one step
alembic downgrade -1

# Then fix the migration file and re-run
alembic upgrade head
```

---

### 🟠 Issue: Frontend Shows Blank Page or Fails to Load

**Symptom:** Browser shows blank white page or network errors.

**Diagnosis:**
```bash
# Check nginx logs
docker compose logs nginx | tail -30

# Check if Vite build exists
ls -la frontend/dist/

# Check browser console for errors (CORS, 404 for JS chunks)
```

**Resolution:**
```bash
# Rebuild frontend
cd frontend && npm run build

# Copy dist to nginx serves directory (or restart nginx container)
docker compose restart frontend
```

If the problem is CORS:
```bash
# Verify CORS_ORIGINS environment variable includes the frontend domain
docker compose exec api env | grep CORS
```

---

### 🟡 Issue: SHAP Setup Failed — No Feature Contributions in Response

**Symptom:** Prediction response returns `top_features: []` (empty).

**Cause:** SHAP TreeExplainer setup failed at startup (non-fatal — predictions still work without SHAP).

**Diagnosis:**
```bash
docker compose logs api | grep -E "(SHAP|shap|explainer)"
```

**Common causes:**
- Dataset file not found (SHAP needs training data to build background)
- SHAP version incompatibility with XGBoost/LightGBM version

**Resolution:**
```bash
# Ensure data files exist
ls data/raw/

# Restart API (SHAP setup runs at startup)
docker compose restart api

# If SHAP still fails, check shap package version
.venv/bin/pip show shap
```

---

## 4. Deployment Procedures

### Standard Deployment

```bash
# 1. Pull latest code
git pull origin main

# 2. Run tests
pytest tests/unit/ -v

# 3. Build new image
docker build -f docker/Dockerfile -t medpredict-api:v1.x.x .

# 4. Run migrations
docker compose -f docker/docker-compose.yml run --rm api alembic upgrade head

# 5. Deploy (zero-downtime rolling update)
docker compose -f docker/docker-compose.yml up -d --no-deps api

# 6. Verify health
curl https://medpredict.yourdomain.com/health

# 7. Monitor logs for 5 minutes
docker compose logs -f api
```

### Emergency Hotfix Deployment

```bash
# Same as standard, but skip extended monitoring
git pull origin hotfix/<name>
docker build -f docker/Dockerfile -t medpredict-api:hotfix .
docker compose -f docker/docker-compose.yml up -d --no-deps api
curl https://medpredict.yourdomain.com/health
```

### Rollback

```bash
# Rollback to previous image
docker tag medpredict-api:v1.x.x-previous medpredict-api:latest
docker compose up -d --no-deps api

# Rollback database migration if needed (use with caution)
alembic downgrade -1
```

---

## 5. Backup & Recovery

### Database Backup

```bash
# Manual backup
pg_dump -h $DB_HOST -U $DB_USER -d medpredict \
  -F c -f "backup_$(date +%Y%m%d_%H%M%S).dump"

# Automated (add to cron or cloud scheduler)
0 2 * * * pg_dump -h $DB_HOST -U $DB_USER medpredict -F c -f /backups/medpredict_$(date +\%Y\%m\%d).dump
```

### Database Restore

```bash
# Stop API first to prevent writes during restore
docker compose stop api

# Restore from dump
pg_restore -h $DB_HOST -U $DB_USER -d medpredict backup.dump

# Restart API
docker compose start api
```

### MLflow Model Recovery

If model files are corrupted:
```bash
# Retrain models from scratch
python scripts/run_training.py --disease all --trials 50

# New model versions will be registered automatically
docker compose restart api
```

---

## 6. Escalation Matrix

| Issue Type | First Responder | Escalate To | Max Response Time |
|---|---|---|---|
| API down | On-call Engineer | Engineering Lead | 30 minutes |
| Data breach | On-call Engineer | Engineering Lead + CISO | 1 hour |
| Patient data exposure | Engineering Lead | Clinical Director + Legal | Immediate |
| Model performance degraded | Engineering Lead | ML Engineer | 4 hours |
| Database corruption | On-call Engineer | Engineering Lead + DBA | 2 hours |
| Auth system down | On-call Engineer | Engineering Lead | 1 hour |

**On-call contacts:**
- Engineering Lead: `engineering-oncall@medpredict.example.com`
- Clinical Director: `clinical@medpredict.example.com`
- Security: `security@medpredict.example.com`
