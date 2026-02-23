#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONTAINER_NAME="streamlit-security-proxy"
CONF_FILE="$ROOT_DIR/ops/nginx/security-proxy.conf"

if [[ ! -f "$CONF_FILE" ]]; then
  echo "[ERROR] Missing proxy config: $CONF_FILE"
  exit 1
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --add-host host.docker.internal:host-gateway \
  -p 8080:8080 \
  -v "$CONF_FILE:/etc/nginx/conf.d/default.conf:ro" \
  nginx:stable >/dev/null

echo "[OK] Security proxy started at http://127.0.0.1:8080"
echo "     (upstream Streamlit target: http://host.docker.internal:8501)"
