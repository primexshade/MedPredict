# Security Policy

## Overview

MedPredict is a clinical decision support tool that handles sensitive medical information. We take security very seriously. This document describes our security practices and the process for reporting vulnerabilities.

---

## Supported Versions

| Version | Security Support |
|---|---|
| 1.0.x | ✅ Active |
| 0.x.x | ❌ End of Life |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security vulnerabilities by emailing:
**security@medpredict.example.com**

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Your suggested fix (optional)

**Response SLA:**
- Acknowledgement: within 48 hours
- Initial assessment: within 5 business days
- Fix timeline (critical): within 7 days
- Fix timeline (high): within 30 days
- Fix timeline (medium/low): within 90 days

---

## Authentication & Authorization

### JWT Token Security

- Tokens are signed with **HS256** (HMAC-SHA256) using a secret key of at minimum 32 characters
- Access tokens expire in **30 minutes**; refresh tokens in **7 days**
- `JWT_SECRET` must never be committed to source control or logged
- In production, rotate `JWT_SECRET` every 90 days; this invalidates all existing sessions
- JTI (JWT ID) field is included to enable token revocation via Redis blacklist

### Password Security

- Passwords are hashed with **bcrypt** (adaptive cost factor), never stored in plaintext
- Minimum password length: 8 characters (configurable)
- Brute-force protection: rate limiting on `/auth/login` (10 attempts/minute per IP)

### Role-Based Access Control

Access is controlled by three roles with escalating permissions:

| Permission | `viewer` | `clinician` | `admin` |
|---|---|---|---|
| View dashboard & analytics | ✅ | ✅ | ✅ |
| Make predictions | ❌ | ✅ | ✅ |
| Read/write patients | ❌ | ✅ | ✅ |
| Delete records | ❌ | ❌ | ✅ |
| Manage users | ❌ | ❌ | ✅ |
| Access audit logs | ❌ | ❌ | ✅ |

---

## Data Protection

### Medical Data Handling

- **No real patient PII is stored in plain text** — patient names and identifiers are stored in PostgreSQL which should be encrypted at rest in production
- The `predictions.input_features` JSONB column stores only the biomarker values, never biometric identifiers
- SHAP values in `predictions.shap_values` are mathematical attributions, not identifiers
- All database connections must use TLS in production (`?sslmode=require` in DATABASE_URL)

### Data in Transit

- All API communication must be over HTTPS in production (nginx terminates TLS)
- Minimum TLS version: TLS 1.2 (TLS 1.3 preferred)
- CORS is configured to only allow specified origins (never `*` in production)

### Data at Rest

- PostgreSQL must have encryption-at-rest enabled (managed DB services: AWS RDS, GCP Cloud SQL provide this by default)
- Redis may contain cached prediction results for 5 minutes; use Redis AUTH and TLS in production
- MLflow model artifacts must be stored on encrypted storage

### Audit Logging

Every sensitive action is recorded in the `audit_logs` table (append-only):
- Login / logout events
- Prediction requests (who ran it, when, for which patient)
- Patient record creation / deletion
- User management actions

Audit logs are never deleted and are excluded from standard backup rotation.

---

## API Security

### Rate Limiting

SlowAPI enforces rate limits per client IP:

| Endpoint Group | Limit |
|---|---|
| `POST /auth/login` | 10/minute |
| `POST /predict/*` | 20/minute |
| All other endpoints | 100/minute |

Exceeding limits returns `429 Too Many Requests` with a `Retry-After` header.

### Input Validation

All API inputs are validated by **Pydantic v2** with strict field ranges:
- Each numeric field has explicit `ge` (≥) and `le` (≤) bounds matching clinical constraints
- String fields are stripped and length-limited
- No raw SQL is ever constructed from user input (SQLAlchemy ORM handles all DB queries)

### CORS

Configure `CORS_ORIGINS` to list only trusted origins:

```bash
# Production example
CORS_ORIGINS=["https://medpredict.yourdomain.com"]

# Never use in production:
CORS_ORIGINS=["*"]
```

### HTTP Headers

The nginx reverse proxy should set these security headers:

```nginx
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options DENY;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';";
add_header Referrer-Policy strict-origin-when-cross-origin;
```

---

## Dependency Security

### Python Dependencies

```bash
# Scan for known vulnerabilities
pip install safety
safety check

# Or with pip-audit
pip install pip-audit
pip-audit
```

### Node.js Dependencies

```bash
cd frontend
npm audit
npm audit fix
```

Dependencies are reviewed on every pull request via GitHub's Dependabot integration.

---

## Infrastructure Security

### Docker

- API container runs as a **non-root user** (`USER medpredict`)
- Docker images use the `slim` base (minimal attack surface)
- No secrets are baked into Docker images — all secrets via environment variables
- Container capabilities are dropped to minimum necessary

### Secrets Management

**Development:** Use `.env` file (in `.gitignore`)

**Staging/Production:** Use a secrets manager:
- AWS: AWS Secrets Manager
- GCP: Secret Manager
- Kubernetes: External Secrets Operator

Never store secrets in:
- Source code
- Docker images
- Kubernetes ConfigMaps (use Secrets instead)
- Log files

### Network

In production Kubernetes:
```yaml
# Restrict inter-pod communication with NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: medpredict-api
spec:
  podSelector:
    matchLabels:
      app: medpredict-api
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx
    ports:
    - port: 8000
```

---

## Incident Response

### Severity Levels

| Level | Examples | Response Time | Notify |
|---|---|---|---|
| **P0 Critical** | Auth bypass, data breach, patient data exposure | Immediate | All stakeholders |
| **P1 High** | Prediction endpoint down, RCE vulnerability | 2 hours | On-call team |
| **P2 Medium** | Rate limit bypass, minor data leak | 24 hours | Engineering lead |
| **P3 Low** | Missing security header, minor info disclosure | 7 days | Next sprint |

### Breach Response Protocol

1. **Contain**: Immediately disable affected endpoints or rotate compromised credentials
2. **Assess**: Determine scope — which data, which patients, what time range
3. **Notify**: Alert clinical leadership within 1 hour for any patient data exposure
4. **Remediate**: Deploy fix (P0: deploy immediately, P1: deploy within 4 hours)
5. **Post-mortem**: Write incident report within 5 business days

For any breach involving real patient data, consult your organization's HIPAA compliance officer within 24 hours.

---

## Security Checklist for Production

Before deploying to production, verify:

- [ ] `JWT_SECRET` is a random 64-character string (not the default)
- [ ] `DATABASE_URL` uses SSL (`?sslmode=require`)
- [ ] Redis has authentication (`requirepass`) enabled
- [ ] HTTPS is enforced (nginx with valid certificate)
- [ ] `CORS_ORIGINS` does not include `*`
- [ ] `ENVIRONMENT=production` is set
- [ ] Docker containers run as non-root
- [ ] All security headers are set in nginx
- [ ] `pip audit` and `npm audit` return no critical vulnerabilities
- [ ] Audit logging is enabled and logs are shipped to a SIEM
- [ ] Backup encryption is enabled for PostgreSQL
