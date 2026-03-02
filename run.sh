#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

echo "[startup] Python: $(python --version 2>&1)"
echo "[startup] Streamlit log level: ${STREAMLIT_LOG_LEVEL:-info}"
echo "[startup] App log level: ${APP_LOG_LEVEL:-INFO}"

# Guard against empty theme overrides from deployment app settings.
# Empty STREAMLIT_THEME_* values can produce browser warnings like:
# "Invalid color passed for ... in theme.sidebar: \"\""
theme_override_count=0
theme_empty_unset_count=0
while IFS='=' read -r key _; do
	if [[ "$key" == STREAMLIT_THEME_* ]]; then
		theme_override_count=$((theme_override_count + 1))
		value="${!key-}"
		if [[ -z "${value//[[:space:]]/}" ]]; then
			unset "$key"
			theme_empty_unset_count=$((theme_empty_unset_count + 1))
			echo "[startup] Unset empty theme override: $key"
		else
			echo "[startup] Theme override: $key=$value"
		fi
	fi
done < <(env)

echo "[startup] Streamlit theme overrides detected: $theme_override_count (empty removed: $theme_empty_unset_count)"

exec python -u -m streamlit run app.py \
	--server.port 8000 \
	--server.address 0.0.0.0 \
	--server.enableCORS false \
	--logger.level "${STREAMLIT_LOG_LEVEL:-info}"