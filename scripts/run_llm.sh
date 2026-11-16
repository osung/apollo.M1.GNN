#!/usr/bin/env bash
# Run a command inside the vLLM docker image with the GPU, the repository and
# the data directory mounted at their host paths (so data/* symlinks resolve).
#
#   scripts/run_llm.sh python3 scripts/llm_match_score.py --direction p2c
#
# GB10 policy (see ~/work/apollo/GB10_LLM_운영가이드.md): LLMs run in docker and
# use at most 85% of unified memory. The scorer takes llm.gpu_mem_util (0.70)
# from config/llm_match.yaml and clamps any override to llm.gpu_mem_util_cap.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LLM_IMAGE:-vllm-apollo:latest}"
DATA_ROOT="$(dirname "$(readlink -f "$ROOT/data/processed")")"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

if docker ps --format '{{.Image}}' | grep -q '^vllm-'; then
    echo "[run_llm] another vLLM container is already running on this host:" >&2
    docker ps --format '  {{.Names}}  {{.Image}}  {{.Status}}' >&2
    if [[ "${ALLOW_CONCURRENT_LLM:-0}" != "1" ]]; then
        echo "[run_llm] refusing to start a second one (unified memory would be oversubscribed)." >&2
        echo "[run_llm] set ALLOW_CONCURRENT_LLM=1 to override." >&2
        exit 1
    fi
fi

exec docker run --rm \
    --gpus all \
    --ipc=host \
    -v "$ROOT":"$ROOT" \
    -v "$DATA_ROOT":"$DATA_ROOT" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -w "$ROOT" \
    "$IMAGE" \
    "$@"
