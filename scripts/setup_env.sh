#!/usr/bin/env bash
# Create the project virtualenv (.venv) and link the data directory.
#
# Usage:
#   scripts/setup_env.sh [DATA_ROOT]
#
# DATA_ROOT (default: ~/data/apollo.data) must contain raw/ with the five
# source pickles. Processed artifacts go to DATA_ROOT/processed so that the
# large files stay outside the repository.
#
# Uses uv with a managed CPython: the system python on the GB10 boxes ships
# without headers, and LightFM has to be compiled from source on aarch64.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${APOLLO_DATA_ROOT:-$HOME/data/apollo.data}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
LIGHTFM_REV="${LIGHTFM_REV:-0c9c31e027b976beab2385e268b58010fff46096}"

cd "$ROOT"

# uv comes from PyPI into a throwaway bootstrap venv (system pip is
# externally managed, and we avoid piping a remote installer into a shell).
BOOTSTRAP="$ROOT/.venv-bootstrap"
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
else
    if [ ! -x "$BOOTSTRAP/bin/uv" ]; then
        echo "[setup] installing uv into $BOOTSTRAP"
        python3 -m venv "$BOOTSTRAP"
        "$BOOTSTRAP/bin/pip" install --quiet uv
    fi
    UV="$BOOTSTRAP/bin/uv"
fi
uv() { "$UV" "$@"; }

# Trust the OS certificate store: the network sits behind a TLS-inspecting
# gateway whose CA is installed system-wide but is unknown to uv's bundled roots.
export UV_SYSTEM_CERTS=1 UV_NATIVE_TLS=1

echo "[setup] creating .venv (python $PYTHON_VERSION)"
uv venv --allow-existing --python "$PYTHON_VERSION" .venv

echo "[setup] installing torch from $TORCH_INDEX_URL"
uv pip install --python .venv/bin/python torch --index-url "$TORCH_INDEX_URL"

echo "[setup] installing requirements"
uv pip install --python .venv/bin/python -r requirements.txt

# The PyPI sdist of lightfm 1.17 fails to build on recent setuptools
# (`__LIGHTFM_SETUP__` AttributeError); the upstream repository is fixed.
echo "[setup] building lightfm from source"
uv pip install --python .venv/bin/python "lightfm @ git+https://github.com/lyst/lightfm.git@$LIGHTFM_REV"

echo "[setup] linking data -> $DATA_ROOT"
mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/raw_merged" "$DATA_ROOT/processed" data
for d in raw raw_merged processed; do
    if [ ! -e "data/$d" ]; then
        ln -s "$DATA_ROOT/$d" "data/$d"
    fi
done
# data/source: DATA_ROOT itself (the *_with_desc pickles used by the LLM matching baseline)
if [ ! -e data/source ]; then
    ln -s "$DATA_ROOT" data/source
fi

.venv/bin/python - <<'EOF'
import faiss, lightfm, torch, torch_geometric
print(f"[setup] torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"pyg {torch_geometric.__version__} lightfm {lightfm.__version__}")
if torch.cuda.is_available():
    print(f"[setup] gpu: {torch.cuda.get_device_name(0)}")
EOF
