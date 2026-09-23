#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    printf 'Сначала выполните: bash install_mac.sh\n'
    exit 1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
printf 'Откройте http://127.0.0.1:8501 в браузере. Для остановки: Ctrl+C.\n'
exec .venv/bin/python server.py --port 8501
