#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import platform, sys
if sys.version_info < (3, 9):
    raise SystemExit('Нужен Python 3.9 или новее.')
print('Система:', platform.system(), platform.machine(), 'Python', platform.python_version())
PY
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade 'pip==25.0.1'
SYSTEM_NAME="$(uname -s)"
MACHINE_NAME="$(uname -m)"
PYTHON_TAG="$(.venv/bin/python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
if [ "$SYSTEM_NAME" = "Darwin" ] && [ "$MACHINE_NAME" = "x86_64" ] && [ "$PYTHON_TAG" = "cp39" ]; then
    .venv/bin/python -m pip install --only-binary=:all: -r requirements-mac.lock
    .venv/bin/python -m pip install 'https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.2/llama_cpp_python-0.3.2-cp39-cp39-macosx_10_9_x86_64.whl'
else
    .venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
    if [ "$SYSTEM_NAME" = "Linux" ]; then
        CC="$(command -v gcc)" CXX="$(command -v g++)" CMAKE_ARGS='-DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF' CMAKE_BUILD_PARALLEL_LEVEL=2 \
            .venv/bin/python -m pip install --no-binary=llama-cpp-python 'llama-cpp-python==0.3.2'
    else
        .venv/bin/python -m pip install 'llama-cpp-python==0.3.2' --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    fi
fi
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > installed_versions.txt
.venv/bin/python doctor.py
printf '\nУстановка завершена. Следующий шаг: .venv/bin/python prepare_models.py\n'
