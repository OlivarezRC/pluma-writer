#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="streamlit-security-proxy"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "[OK] Security proxy stopped"
