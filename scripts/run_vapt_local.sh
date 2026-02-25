#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_URL="${1:-http://127.0.0.1:8501}"

echo "============================================================"
echo "Local VAPT Baseline"
echo "Repo: $ROOT_DIR"
echo "DAST Target: $TARGET_URL"
echo "============================================================"

if [[ ! -f requirements.txt ]]; then
  echo "[ERROR] requirements.txt not found. Run from repository root."
  exit 1
fi

echo "[1/5] Installing lightweight security tooling..."
python -m pip install --upgrade pip >/dev/null
python -m pip install pip-audit bandit detect-secrets >/dev/null

echo "[2/5] Secrets baseline scan (working tree)..."
detect-secrets scan --all-files --exclude-files '^(venv|\.venv|__pycache__|\.git|data/).*' > .secrets.baseline.local
echo "  - Wrote .secrets.baseline.local"

echo "[3/5] Static security scan (Bandit)..."
bandit -r app deep_research pages app.py -x __pycache__,venv,.venv || true

echo "[4/5] Dependency CVE scan (pip-audit)..."
pip-audit -r requirements.txt || true

echo "[5/5] Optional DAST baseline via OWASP ZAP (requires running app + Docker)..."
if command -v docker >/dev/null 2>&1; then
  docker run --rm -t --add-host host.docker.internal:host-gateway -v "$PWD:/zap/wrk:rw" ghcr.io/zaproxy/zaproxy:stable \
    zap-baseline.py -t "$TARGET_URL" -r zap_report.html || true
  if [[ -f zap_report.html ]]; then
    echo "  - Wrote zap_report.html"
  else
    echo "  - No zap_report.html generated (target may be unreachable or scan failed)"
  fi
else
  echo "  - Skipped: docker not available"
fi

echo "============================================================"
echo "VAPT baseline run completed."
echo "Review: Bandit output, pip-audit output, .secrets.baseline.local, zap_report.html"
echo "============================================================"
