#!/usr/bin/env bash
set -Eeuo pipefail

readonly MODEL_NAME="gemma4:26b"
readonly OLLAMA_API_BASE="http://127.0.0.1:11434"
readonly STARTUP_ATTEMPTS=60
server_pid=""

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

cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        printf '\nStopping Ollama server...\n'
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
}

server_is_ready() {
    curl --fail --silent --show-error \
        --connect-timeout 1 \
        --max-time 2 \
        "$OLLAMA_API_BASE/api/tags" >/dev/null 2>&1
}

wait_for_server() {
    local attempt server_status

    for ((attempt = 1; attempt <= STARTUP_ATTEMPTS; attempt++)); do
        if server_is_ready; then
            return
        fi

        if ! kill -0 "$server_pid" 2>/dev/null; then
            server_status=0
            wait "$server_pid" || server_status=$?
            fail "Ollama server exited during startup with status $server_status."
        fi
        sleep 1
    done

    fail "Ollama did not become ready at $OLLAMA_API_BASE within $STARTUP_ATTEMPTS seconds."
}

check_cuda_gpu() {
    local gpu_list

    if ! gpu_list="$(nvidia-smi -L 2>/dev/null)"; then
        fail "The NVIDIA driver could not communicate with a CUDA-capable GPU."
    fi
    [[ -n "$gpu_list" ]] || fail "No CUDA-capable NVIDIA GPU was detected."

    printf 'Detected NVIDIA GPU:\n'
    printf '  %s\n' "${gpu_list%%$'\n'*}"
}

preload_model() {
    printf 'Pulling %s if needed...\n' "$MODEL_NAME"
    ollama pull "$MODEL_NAME"

    printf 'Loading %s on the GPU...\n' "$MODEL_NAME"
    curl --fail --silent --show-error \
        --max-time 1800 \
        --header 'Content-Type: application/json' \
        --data '{"model":"gemma4:26b","stream":false,"keep_alive":-1}' \
        "$OLLAMA_API_BASE/api/generate" >/dev/null
}

check_model_uses_gpu() {
    local model_line running_models

    if ! running_models="$(ollama ps 2>&1)"; then
        fail "Could not inspect the running Ollama model."
    fi
    model_line="$(awk -v model="$MODEL_NAME" '$1 == model { print; exit }' \
        <<<"$running_models")"

    [[ -n "$model_line" ]] || \
        fail "$MODEL_NAME was pulled but is not loaded."
    [[ "$model_line" == *GPU* ]] || \
        fail "$MODEL_NAME is not using the CUDA GPU: $model_line"

    printf 'Loaded model:\n  %s\n' "$model_line"
}

[[ "$(uname -s)" == "Linux" ]] || \
    fail "This Ollama launcher requires Linux."
require_command ollama "Install it with: bash ./scripts/install-ollama.sh"
require_command curl "Install curl with the system package manager."
require_command nvidia-smi "Install a compatible NVIDIA driver."
check_cuda_gpu

# Keep the CLI and the notebook on the same loopback endpoint.
export OLLAMA_HOST="127.0.0.1:11434"

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if server_is_ready; then
    printf 'Using the Ollama server already running at %s.\n' "$OLLAMA_API_BASE"
else
    printf 'Starting Ollama server at %s...\n' "$OLLAMA_API_BASE"
    ollama serve &
    server_pid=$!
    wait_for_server
fi

preload_model
check_model_uses_gpu

printf '\nOllama is ready for notebooks/local-llm-ollama.ipynb.\n'
if [[ -z "$server_pid" ]]; then
    exit 0
fi

printf 'Keep this terminal open; press Ctrl+C to stop the server.\n'
wait "$server_pid"
