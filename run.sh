#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

echo "[startup] Python: $(python --version 2>&1)"
echo "[startup] Streamlit log level: ${STREAMLIT_LOG_LEVEL:-info}"
echo "[startup] App log level: ${APP_LOG_LEVEL:-INFO}"

exec python -u -m streamlit run app.py \
	--server.port 8000 \
	--server.address 0.0.0.0 \
	--server.enableCORS false \
	--logger.level "${STREAMLIT_LOG_LEVEL:-info}"