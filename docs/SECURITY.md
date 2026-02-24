# Security Model — PneumoDetect AI

> **Document Status:** Production · **Version:** 1.0 · **Classification:** Public

---

## 1. Security Philosophy

PneumoDetect AI is built on three core security principles:

1. **Zero Data Persistence** — Medical images are processed entirely in-memory and never written to disk or any database. No PHI is retained.
2. **Defense in Depth** — Multiple independent security controls at each layer (transport, application, data).
3. **Minimal Attack Surface** — No database, no user accounts, no session state. Stateless by design.

---

## 2. Threat Model

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Malicious file upload (image bomb) | Medium | High | PIL.Image.verify() + size limit (10MB) |
| DDoS / abuse | Medium | Medium | Rate limiting (100 req/min/IP) |
| Secrets exposure | Low | Critical | GitHub Secrets + no-commit enforcement via `.gitignore` |
| Model inversion attack | Low | Low | No training data stored or accessible via API |
| Container escape | Low | High | Non-root user, minimal base image |
| Dependency supply chain | Medium | High | `pip-audit` on every CI run |

---

## 3. Security Controls by Layer

### 3.1 Transport Security

- **TLS 1.3** enforced on all production endpoints (handled by CDN/load balancer)
- **HTTPS-only**: HTTP requests are redirected to HTTPS
- **HSTS headers** applied at the CDN layer

### 3.2 Input Validation

```python
# Applied in api/main.py on every /predict request
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/dicom"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

async def validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Invalid file type")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File too large")
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()  # Detects malformed/corrupt images
    except Exception:
        raise HTTPException(422, "Unprocessable image")
    return contents
```

### 3.3 Application Security

| Control | Implementation |
|---|---|
| **CORS** | `CORSMiddleware` with explicit allowed origins |
| **Rate Limiting** | `slowapi` — 100 requests/min per IP |
| **Error Handling** | Never expose stack traces in public responses |
| **Dependency Pinning** | All versions pinned in `requirements.txt` |

### 3.4 Container Security

```dockerfile
# infra/docker/Dockerfile — security hardening
FROM python:3.11-slim                  # Minimal base image

RUN groupadd -r appuser && \
    useradd -r -g appuser appuser      # Non-root user

COPY --chown=appuser:appuser . /app
USER appuser                           # Switch before CMD

HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
```

### 3.5 Secrets Management

Secrets are **never** committed to the repository. The `.gitignore` enforces exclusion of:

```
.env
.env.*
secrets.toml
*.key
*.pem
credentials.json
```

All runtime secrets are injected via:
- **GitHub Actions Secrets** → CI/CD pipelines
- **Streamlit Cloud Secrets** → Web app deployment
- **Environment Variables** → Docker containers

---

## 4. Data Privacy

| Principle | Implementation |
|---|---|
| **No storage** | Uploaded images processed in RAM only; garbage collected after response |
| **No logging of image content** | Only filename, size, and inference metadata are logged |
| **No user tracking** | No cookies, no analytics, no user accounts |
| **GDPR compatible** | No personal data collected or processed beyond the image itself |

> ⚠️ **Important:** Do NOT upload real patient X-rays to the public demo. The tool is for educational and research use only.

---

## 5. Dependency Scanning

Run on every CI pipeline:

```bash
pip-audit --requirement requirements.txt
```

Critical and high-severity vulnerabilities will block the CI pipeline.

---

## 6. Incident Response

1. **Detection**: Monitoring alerts via uptime checks + Grafana alerting
2. **Triage**: Identify affected component; assess data exposure (none expected)
3. **Containment**: Take service offline if necessary (Render/Streamlit dashboard)
4. **Remediation**: Patch, rebuild container, re-deploy
5. **Post-mortem**: Document in GitHub Issues with label `security`

**Vulnerability disclosure:** See [`SECURITY.md`](../SECURITY.md) in the repository root.

---

## 7. Security Checklist (Pre-Release)

- [x] No secrets committed to repository
- [x] Input validation on all file uploads
- [x] Container running as non-root user
- [x] All dependencies pinned and auditable
- [x] HTTPS enforced on production endpoints
- [x] Zero image data persistence
- [x] Rate limiting configured
- [x] Error messages sanitized (no stack traces in production responses)
