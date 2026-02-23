# VAPT Execution Report (What We Ran, What We Found, What We Fixed)

Date: 2026-02-23  
Repository: `fernandezbr/pluma-writer`  
Branch: `replace-main`

## 1) Objective

This document summarizes the exact VAPT baseline process executed in this repository, the issues detected, and the remediations applied during this hardening cycle.

## 2) Exact VAPT flow used

### Baseline local scan command

```bash
bash scripts/run_vapt_local.sh http://host.docker.internal:8501
```

What this script runs:

1. `detect-secrets` (working tree baseline)
2. `bandit` (SAST)
3. `pip-audit` (dependency CVEs)
4. `OWASP ZAP baseline` via Docker (DAST)

Script reference: `scripts/run_vapt_local.sh`.

### Security-header hardening scan path (recommended path we also used)

```bash
bash scripts/start_secure_proxy.sh
bash scripts/run_vapt_local.sh http://host.docker.internal:8080
bash scripts/stop_secure_proxy.sh
```

Proxy reference: `ops/nginx/security-proxy.conf` (injects CSP and browser security headers).

## 3) Artifacts produced

- `.secrets.baseline.local`
- `zap_report.html`
- Console output for Bandit and pip-audit

## 4) Findings observed

## 4.1 DAST (OWASP ZAP)

From `zap_report.html` (latest artifact):

- High: **0**
- Medium: **3**
- Low: **2**
- Informational: **3**

Main alert families:

- `CSP: script-src unsafe-eval` (Medium)
- `CSP: script-src unsafe-inline` (Medium)
- `CSP: style-src unsafe-inline` (Medium)
- `Dangerous JS Functions` (Low)
- `Timestamp Disclosure - Unix` (Low)
- `Information Disclosure - Suspicious Comments` (Informational)
- `Modern Web Application` (Informational)
- `Non-Storable Content` (Informational, systemic)

Observed trend during hardening reruns in this session:

- Warning count reduced from **11** to **6** after introducing/tightening proxy security headers.

## 4.2 SAST (Bandit)

Key issue classes observed during this hardening cycle:

- Weak hash usage (`hashlib.md5`) → High
- Request without timeout (`requests.get`) → Medium
- Broad `try/except` patterns (`pass` / `continue`) → Low
- SQL-expression heuristic findings in query construction (`B608`) → Medium/Low confidence

Latest tracked session state after fixes:

- High: **0**
- Medium: **5**
- Low: **10**

## 4.3 Dependency scan (pip-audit)

- No known vulnerabilities found in the latest runs.

## 4.4 Secret scan (detect-secrets)

- Baseline file generated successfully.
- Operational action still required if any real credentials were ever exposed: rotate/revoke and move to secret manager.

## 5) Remediations implemented

## 5.1 Code-level remediations

1. Replaced weak hashing with SHA-256 in active paths:
   - `app/writer_main.py`
   - `app/pipeline_enhancements.py`
   - `app/plagiarism_checker.py`

2. Added outbound HTTP timeout:
   - `app/utils.py` now uses `requests.get(..., timeout=10)`.

3. Streamlit security flags enabled:
   - `.streamlit/config.toml`
   - `enableXsrfProtection = true`
   - `enableCORS = true`

## 5.2 DAST/environment remediations

1. Added local security proxy operations:
   - `scripts/start_secure_proxy.sh`
   - `scripts/stop_secure_proxy.sh`

2. Added hardened Nginx header policy:
   - `ops/nginx/security-proxy.conf`
   - Includes `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`, `COOP/COEP/CORP`, cache controls, and CSP.

3. Improved local DAST reliability:
   - `scripts/run_vapt_local.sh` mounts workspace into ZAP container and writes `zap_report.html` reliably.

## 5.3 Process/CI remediations

1. Added pre-commit security baseline:
   - `.pre-commit-config.yaml` with Bandit, detect-secrets, and pip-audit checks.

2. Added CI security workflow:
   - `.github/workflows/security-vapt.yml`
   - Runs Bandit, pip-audit, detect-secrets on push/PR.

## 6) Items intentionally not fully eliminated yet

1. CSP `unsafe-inline` / `unsafe-eval` warnings:
   - Currently retained for Streamlit runtime compatibility in this setup.

2. Some Bandit medium/low findings:
   - Mainly broad exception handling and SQL-string heuristics.
   - Require targeted refactors to reduce while preserving behavior.

3. Secret hygiene operations:
   - Credential rotation/revocation is an operational step outside code-only fixes.

## 7) Current security posture (summary)

- High-severity Bandit findings that were in active code paths were removed.
- Dependency CVE scan is currently clean.
- DAST findings were reduced materially through header hardening.
- Remaining findings are mostly framework/runtime tradeoffs and low/medium hygiene items suitable for iterative cleanup.

## 8) Recommended next pass

1. Add a strict CSP profile (without `unsafe-eval`) as an optional test mode and validate app behavior.
2. Refactor selected broad exception blocks to narrower exception handling + logging.
3. Rework flagged SQL query construction to reduce heuristic hits where feasible.
4. Complete credential rotation and externalize secrets.
