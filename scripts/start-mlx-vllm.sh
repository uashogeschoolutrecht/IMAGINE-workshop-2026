#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-mlx-community/gemma-4-e4b-it-4bit}"
VLLM_BIN="${VLLM_BIN:-$HOME/.venv-vllm-metal/bin/vllm}"

if [[ ! -x "$VLLM_BIN" ]]; then
    printf 'vLLM Metal is not installed at %s. Run ./scripts/install-mlx-vllm.sh first.\n' \
        "$VLLM_BIN" >&2
    exit 1
fi

exec "$VLLM_BIN" serve "$MODEL_NAME"
