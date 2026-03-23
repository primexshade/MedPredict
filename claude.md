# Claude AI Memory — MedPredict Project

> This file serves as persistent memory for Claude AI sessions working on the MedPredict project. It contains critical context, conventions, and lessons learned.

---

## Project Identity

- **Name**: MedPredict
- **Type**: Clinical AI Disease Prediction Platform
- **Diseases**: Heart, Diabetes (Type 2), Breast Cancer, Chronic Kidney Disease
- **Stack**: FastAPI + React + XGBoost/LightGBM + PostgreSQL + Redis

---

## Critical Files

### Backend Entry Points
- `src/api/main.py` — FastAPI app factory, lifespan events
- `src/api/routers/predict.py` — Core prediction endpoints (largest file ~600 LOC)
- `src/api/routers/auth.py` — Authentication (login, refresh, logout)
- `src/api/deps.py` — Shared dependencies (JWT auth, Redis, DB session)
- `src/config.py` — Pydantic Settings (env-driven configuration)

### ML Pipeline
- `src/models/train.py` — Optuna + nested CV + MLflow logging
- `src/features/engineering.py` — Disease-specific feature engineering
- `src/features/pipeline.py` — sklearn ColumnTransformer factory
- `src/scoring/risk_scorer.py` — Composite risk scoring + confidence intervals
- `src/explainability/shap_explainer.py` — SHAP TreeExplainer wrapper

### Frontend Entry Points
- `frontend/src/App.tsx` — Router setup, auth logout handler
- `frontend/src/services/api.ts` — Axios client with interceptors
- `frontend/src/pages/Predict.tsx` — Disease prediction forms (largest ~400 LOC)
- `frontend/src/pages/Dashboard.tsx` — KPI cards, charts

---

## Code Conventions

### Python (Backend)
- **Python version**: 3.11+
- **Formatter**: Black (line-length 100)
- **Linter**: Ruff (strict mode)
- **Type hints**: Required for all public functions
- **Docstrings**: Google style
- **Async**: Use `async def` for all endpoints
- **Logging**: Use `structlog` for structured JSON logs

### TypeScript (Frontend)
- **React**: Functional components with hooks
- **State**: TanStack React Query for server state
- **Styling**: Inline styles with CSS-in-JS objects
- **Types**: Strict TypeScript, no `any` except escape hatches
- **Imports**: Prefer named imports

### Naming Conventions
- **Files**: snake_case (Python), PascalCase (React components)
- **Variables**: snake_case (Python), camelCase (TypeScript)
- **Constants**: SCREAMING_SNAKE_CASE
- **API routes**: kebab-case (`/comorbidity-rules`)

---

## Database Schema

### Core Tables
1. **users** — id (UUID), email, hashed_password, role, is_active
2. **patients** — id (UUID), mrn, dob, sex, primary_clinician_id
3. **predictions** — id (UUID), patient_id, disease, probability, risk_category, shap_json
4. **audit_logs** — id (INT), user_id, action, resource, ip_address

### Relationships
- User 1:N Patient (clinician → assigned patients)
- Patient 1:N Prediction (patient → prediction history)
- User 1:N AuditLog (user → activity log)

---

## Authentication Flow

1. **Login**: POST `/api/v1/auth/login` → Returns `access_token` + `refresh_token`
2. **Token storage**: localStorage (access_token, refresh_token)
3. **Request auth**: Axios interceptor adds `Authorization: Bearer {token}`
4. **Token refresh**: POST `/api/v1/auth/refresh` with refresh_token
5. **Logout**: POST `/api/v1/auth/logout` → Token JTI added to Redis blacklist
6. **401 handling**: Clear tokens, dispatch custom event, navigate to /login

### Default Admin (First Run)
- Email: `admin@medpredict.local`
- Password: `changeme123`
- Created automatically if no users exist

---

## Risk Categories

```python
RISK_THRESHOLDS = {
    "LOW": (0.00, 0.20),
    "BORDERLINE": (0.21, 0.40),
    "MODERATE": (0.41, 0.60),
    "HIGH": (0.61, 0.80),
    "CRITICAL": (0.81, 1.00),
}
```

---

## Common Commands

```bash
# Start backend
cd "Disease Prediction"
source .venv/bin/activate
uvicorn src.api.main:app --reload --port 8000

# Start frontend
cd frontend && npm run dev

# Run tests
pytest tests/ -v --cov=src

# Build frontend
cd frontend && npm run build

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"
```

---

## Known Issues & Workarounds

### Issue: Wrong API running on port 8000
**Symptom**: Health check returns X-IDS Inference API instead of Disease Prediction API
**Cause**: Another uvicorn process running from different project
**Fix**: `lsof -i :8000` to find PID, then `kill <PID>`

### Issue: Database connection errors
**Symptom**: `asyncpg.CannotConnectNowError`
**Cause**: PostgreSQL not running or wrong credentials
**Fix**: Start PostgreSQL, check DATABASE_URL in .env

### Issue: Redis not available
**Symptom**: "Redis not available — using no-op cache stub"
**Cause**: Redis server not running
**Impact**: Caching disabled, but app still works
**Fix**: Start Redis or ignore for local dev

### Issue: MLflow model loading fails
**Symptom**: 503 Service Unavailable on prediction endpoints
**Cause**: Models not trained or mlruns directory missing
**Fix**: Run `python scripts/run_training.py --disease all`

---

## Testing Strategy

### Unit Tests
- Feature engineering functions
- Risk scoring logic
- Token creation/validation

### Integration Tests
- Auth flow (login → protected endpoint → logout)
- Prediction endpoints with mocked models
- Database CRUD operations

### E2E Tests
- Full prediction workflow
- Patient management

---

## Performance Notes

- **Prediction latency target**: <200ms p99
- **Model loading**: ~2-3 seconds per model at startup
- **Redis caching**: 5-minute TTL on prediction results
- **Rate limits**: 100 req/min general, 20 req/min predictions

---

## Security Checklist

- [x] bcrypt password hashing (cost factor 12)
- [x] JWT with configurable expiry
- [x] Token blacklist for logout
- [x] RBAC with permission checks
- [x] Rate limiting
- [x] Input validation (Pydantic)
- [x] SQL injection prevention (ORM)
- [x] CORS configuration
- [ ] HTTPS enforcement (deployment-time)
- [ ] CSRF protection (not needed with JWT)

---

## Deployment Checklist

1. Set `ENVIRONMENT=production`
2. Generate strong `JWT_SECRET` (32+ chars)
3. Configure PostgreSQL with connection pooling
4. Enable Redis for caching
5. Set `CORS_ORIGINS` to production domain
6. Configure reverse proxy (nginx) with HTTPS
7. Set up log aggregation
8. Configure health check monitoring

---

## Useful Debugging Commands

```bash
# Check API health
curl http://localhost:8000/health

# Get OpenAPI schema
curl http://localhost:8000/openapi.json | jq

# Test login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@medpredict.local","password":"changeme123"}'

# Test prediction (with token)
curl -X POST http://localhost:8000/api/v1/predict/heart \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":140,...}'
```

---

## Recent Changes (Bugfix Branch)

### Security Fixes Applied
1. Removed hardcoded `admin@example.com / admin` credentials
2. Implemented proper database user authentication
3. Added `Depends(get_current_user)` to all protected endpoints
4. Implemented logout with Redis token blacklist
5. Fixed Redis pool race condition with `asyncio.Lock`

### Frontend Fixes Applied
1. Changed `window.location.href` to React Router `navigate()`
2. Added proper error handling to all API calls
3. Fixed React key warnings in Recharts
4. Added TypeScript error types

---

## Contact & Resources

- **API Docs**: http://localhost:8000/docs
- **Project Documentation**: `/docs/PROJECT_DOCUMENTATION.md`
- **MLflow UI**: http://localhost:5001 (when running)

---

*Last updated: March 2026*
