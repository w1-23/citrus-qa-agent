#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$APP_ROOT/.." && pwd)"
DATA_ROOT="$ROOT/agent"

mkdir -p "$DATA_ROOT/state" "$DATA_ROOT/logs" "$DATA_ROOT/workspace" "$ROOT/tmp"
chmod 700 "$DATA_ROOT/state" "$DATA_ROOT/logs" "$ROOT/tmp"

# Keep caches and temporary uploads on the data volume. One worker is deliberate:
# the embedded retrieval models are process singletons and should be loaded once.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export TMPDIR="$ROOT/tmp"
export TEMP="$ROOT/tmp"
export TMP="$ROOT/tmp"
export XDG_CACHE_HOME="$DATA_ROOT/.cache"
export HF_HOME="$DATA_ROOT/.hf_cache/hf"
export CITRUSQA_MODEL_CACHE_ROOT="$DATA_ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export FASTEMBED_CACHE_PATH="$DATA_ROOT/.hf_cache/fastembed"
export CITRUSQA_CPU_THREADS="${CITRUSQA_CPU_THREADS:-8}"
export PYTHONPATH="$APP_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$APP_ROOT"
exec "$ROOT/.venv/bin/python" -m uvicorn src.api.main:app \
  --host "${CITRUSQA_HOST:-127.0.0.1}" \
  --port "${CITRUSQA_PORT:-8000}" \
  --proxy-headers \
  --forwarded-allow-ips="${CITRUSQA_TRUSTED_PROXY_IPS:-127.0.0.1}" \
  --workers 1
