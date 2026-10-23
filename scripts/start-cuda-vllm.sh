#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly VENV_DIR="${VLLM_VENV:-$PROJECT_DIR/.venv-vllm}"
readonly DEFAULT_MODEL="google/gemma-4-E4B-it"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ "$(uname -s)" == "Linux" ]] || \
    fail "CUDA vLLM can only be started by this script on Linux."
command -v uv >/dev/null 2>&1 || \
    fail "uv is required. Run ./scripts/install-python.sh first."
[[ -x "$VENV_DIR/bin/python" ]] || \
    fail "The local vLLM environment is missing. Run ./scripts/install-cuda-vllm.sh first."

if (($# == 0)) || [[ "$1" == -* ]]; then
    set -- "$DEFAULT_MODEL" "$@"
fi

if ! installed_version="$(uv run --no-project --python "$VENV_DIR/bin/python" -- \
    python -c 'import vllm; print(vllm.__version__)')"; then
    fail "vLLM cannot be imported from $VENV_DIR. Run ./scripts/install-cuda-vllm.sh again."
fi

if ! cuda_device="$(uv run --no-project --python "$VENV_DIR/bin/python" -- \
    python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")

device = torch.device("cuda:0")
value = torch.ones(1, device=device)
if value.item() != 1:
    raise SystemExit("CUDA computation returned an unexpected result")
torch.cuda.synchronize(device)
print(torch.cuda.get_device_name(device))
PY
)"; then
    fail "The vLLM environment cannot run a CUDA computation. Run ./scripts/install-cuda-vllm.sh again."
fi

printf 'Starting vLLM %s on %s...\n' "$installed_version" "$cuda_device"
exec uv run --no-project --python "$VENV_DIR/bin/python" -- vllm serve "$@"
