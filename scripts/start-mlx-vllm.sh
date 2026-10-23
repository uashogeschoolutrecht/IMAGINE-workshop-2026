#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
    printf 'vLLM Metal requires native Apple Silicon macOS.\n' >&2
    exit 1
}
readonly VLLM_BIN="${VLLM_BIN:-$HOME/.venv-vllm-metal/bin/vllm}"
readonly DEFAULT_MODEL="mlx-community/gemma-4-e4b-it-4bit"

if [[ ! -x "$VLLM_BIN" ]]; then
    printf 'vLLM Metal is not installed at %s. Run ./scripts/install-mlx-vllm.sh first.\n' \
        "$VLLM_BIN" >&2
    exit 1
fi
if (($# == 0)) || [[ "$1" == -* ]]; then
    set -- "$DEFAULT_MODEL" "$@"
fi
exec "$VLLM_BIN" serve "$@"
