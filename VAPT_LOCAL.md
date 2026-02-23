# Local VAPT Baseline

This repository includes a practical VAPT baseline to find issues early during development.

## What is covered

- SAST (Bandit)
- Dependency vulnerabilities (pip-audit)
- Secret discovery (detect-secrets)
- Optional local DAST baseline (OWASP ZAP)

## 1) Run locally

From repo root:

```bash
bash scripts/run_vapt_local.sh
```

Optional target URL for ZAP baseline:

```bash
bash scripts/run_vapt_local.sh http://127.0.0.1:8501
```

Outputs:

- `.secrets.baseline.local`
- `zap_report.html` (if Docker + reachable target)
- console output for Bandit and pip-audit

## 2) Pre-commit security checks

Install and enable:

```bash
pip install pre-commit pip-audit
detect-secrets scan --all-files > .secrets.baseline
pre-commit install
```

Run manually:

```bash
pre-commit run --all-files
```

## 3) CI workflow

Workflow file:

- `.github/workflows/security-vapt.yml`

Runs on push/PR and checks:

- Bandit
- pip-audit
- detect-secrets baseline generation

## 4) Optional security reverse-proxy (recommended for header hardening checks)

When scanning Streamlit directly, ZAP typically reports header warnings (CSP, X-Content-Type-Options, anti-clickjacking, etc.).
Use the local Nginx security proxy to inject hardening headers before DAST:

```bash
bash scripts/start_secure_proxy.sh
bash scripts/run_vapt_local.sh http://host.docker.internal:8080
```

Stop proxy when done:

```bash
bash scripts/stop_secure_proxy.sh
```

## Recommended remediation process

1. Triage findings by severity and exploitability.
2. Fix high/critical first (secrets, auth flaws, injection risks, vulnerable packages).
3. Re-run local baseline (`run_vapt_local.sh`).
4. Open PR with remediation notes and evidence.
5. Ensure CI workflow passes before merge.

## Important

If any real credentials were committed or exposed, rotate/revoke them immediately and replace with secrets manager-backed configuration.
