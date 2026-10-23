#!/usr/bin/env bash
set -Eeuo pipefail

# Install the CUDA build of vLLM in a repository-local, uv-managed virtual
# environment. This keeps vLLM and its Python version separate from the
# workshop's Python 3.14 environment.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly VENV_DIR="${VLLM_VENV:-$PROJECT_DIR/.venv-vllm}"
readonly PYTHON_VERSION="${VLLM_PYTHON:-3.12}"
readonly VLLM_PACKAGE="${VLLM_PACKAGE:-vllm}"
readonly TORCH_BACKEND="${VLLM_TORCH_BACKEND:-auto}"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    local command_name="$1"
    local install_hint="$2"

    command -v "$command_name" >/dev/null 2>&1 || \
        fail "Required command '$command_name' is missing. $install_hint"
}

check_cuda_installation() {
    local cuda_version gpu_list nvcc_output

    require_command nvidia-smi \
        "Install a compatible NVIDIA driver before installing vLLM."
    require_command nvcc \
        "Install the NVIDIA CUDA Toolkit and make its bin directory available on PATH."

    if ! nvcc_output="$(nvcc --version 2>&1)"; then
        fail "nvcc is present but the CUDA Toolkit is not working."
    fi
    if [[ "$nvcc_output" =~ release[[:space:]]+([0-9]+\.[0-9]+) ]]; then
        cuda_version="${BASH_REMATCH[1]}"
    else
        fail "Could not determine the CUDA Toolkit version from 'nvcc --version'."
    fi

    if ! gpu_list="$(nvidia-smi -L 2>/dev/null)"; then
        fail "The NVIDIA driver could not communicate with a CUDA-capable GPU."
    fi
    [[ -n "$gpu_list" ]] || fail "No CUDA-capable NVIDIA GPU was detected."

    printf 'Detected CUDA Toolkit %s and NVIDIA GPU:\n' "$cuda_version"
    printf '  %s\n' "${gpu_list%%$'\n'*}"
}

check_installed_cuda_runtime() {
    local runtime_summary

    if ! runtime_summary="$(uv run --no-project --python "$VENV_DIR/bin/python" -- \
        python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")

device_index = 0
capability = torch.cuda.get_device_capability(device_index)
if capability < (7, 5):
    raise SystemExit(
        f"vLLM requires compute capability 7.5 or newer; GPU 0 has "
        f"{capability[0]}.{capability[1]}"
    )

value = torch.ones(1, device=f"cuda:{device_index}")
if value.item() != 1:
    raise SystemExit("CUDA computation returned an unexpected result")
torch.cuda.synchronize(device_index)

print(
    f"{torch.cuda.get_device_name(device_index)} "
    f"(compute capability {capability[0]}.{capability[1]}, "
    f"PyTorch CUDA {torch.version.cuda})"
)
PY
    )"; then
        fail "vLLM was installed, but its PyTorch environment could not run a CUDA computation."
    fi

    printf 'CUDA runtime check passed: %s\n' "$runtime_summary"
}

[[ "$(uname -s)" == "Linux" ]] || \
    fail "vLLM can only be installed by this script on Linux."
require_command uv "Run ./scripts/install-python.sh first."
check_cuda_installation

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if [[ -e "$VENV_DIR" ]]; then
        fail "$VENV_DIR exists but is not a usable virtual environment."
    fi

    printf 'Creating local vLLM environment at %s...\n' "$VENV_DIR"
    uv venv --python "$PYTHON_VERSION" --managed-python "$VENV_DIR"
fi

actual_python="$("$VENV_DIR/bin/python" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$actual_python" != "$PYTHON_VERSION" ]]; then
    fail "$VENV_DIR uses Python $actual_python; vLLM expects Python $PYTHON_VERSION. Remove the environment and rerun this script."
fi

printf 'Installing %s with PyTorch backend %s...\n' "$VLLM_PACKAGE" "$TORCH_BACKEND"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    --upgrade \
    --torch-backend "$TORCH_BACKEND" \
    "$VLLM_PACKAGE"

installed_version="$(uv run --no-project --python "$VENV_DIR/bin/python" -- \
    python -c 'from importlib.metadata import version; print(version("vllm"))')"
check_installed_cuda_runtime

printf 'Installed vLLM %s in %s.\n' "$installed_version" "$VENV_DIR"
printf 'Start it through uv with:\n'
printf '  %s/start-cuda-vllm.sh\n' "$SCRIPT_DIR"
