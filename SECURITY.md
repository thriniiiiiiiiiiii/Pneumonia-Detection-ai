# Security Policy — PneumoDetect AI

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Yes |
| < 1.0   | ❌ No  |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities via public GitHub Issues.**

If you discover a security vulnerability, please report it responsibly:

1. **Email**: [ayushirathour1804@gmail.com](mailto:ayushirathour1804@gmail.com)
2. **Subject line**: `[SECURITY] PneumoDetect AI — <brief description>`
3. **Include**:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Any suggested remediation (optional)

### Response Timeline

| Step | Timeline |
|---|---|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 5 business days |
| Patch release (critical) | Within 14 days |
| Patch release (non-critical) | Within 30 days |
| Public disclosure | After patch is released |

## Security Scope

### In Scope

- Arbitrary file read/write via the `/predict` endpoint
- Remote code execution via malicious image files
- Authentication bypass (if auth is added)
- Dependency vulnerabilities (CVE-rated critical or high)
- Sensitive data exposure

### Out of Scope

- Vulnerabilities in third-party services (HuggingFace, Render, Streamlit Cloud)
- Social engineering attacks
- Denial of service attacks on the public demo (rate-limiting is best-effort)
- Issues in browsers or operating systems

## Disclosure Policy

We follow [Coordinated Vulnerability Disclosure (CVD)](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). We will credit responsible reporters in the release notes (with your consent).

---

*This security policy is reviewed and updated annually.*
