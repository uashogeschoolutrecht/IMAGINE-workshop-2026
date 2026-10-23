#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
    printf 'vLLM Metal requires native Apple Silicon macOS.\n' >&2
    exit 1
}
command -v curl >/dev/null 2>&1 || { printf 'Install curl first.\n' >&2; exit 1; }

readonly VENV="$HOME/.venv-vllm-metal"
readonly INSTALLER_URL="https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh"

installed_version() {
    if [[ ! -x "$VENV/bin/python" ]]; then
        return 1
    fi

    "$VENV/bin/python" -c \
        'from importlib.metadata import version; print(version("vllm-metal"))' \
        2>/dev/null
}

before_version="$(installed_version || true)"

if [[ -n "$before_version" ]]; then
    printf 'Checking for vllm-metal updates (installed: %s)...\n' "$before_version"
else
    printf 'Installing vllm-metal into %s...\n' "$VENV"
fi

# This is the default installation method recommended by vllm-metal. The
# upstream stable installer creates the environment and updates it on later runs.
curl -fsSL "$INSTALLER_URL" | bash -s -- --stable

after_version="$(installed_version || true)"
if [[ -z "$after_version" ]]; then
    printf 'Error: vllm-metal was not found in %s after installation.\n' "$VENV" >&2
    exit 1
fi

if [[ -z "$before_version" ]]; then
    printf 'Installed vllm-metal %s.\n' "$after_version"
elif [[ "$before_version" == "$after_version" ]]; then
    printf 'vllm-metal is already up to date (%s).\n' "$after_version"
else
    printf 'Updated vllm-metal from %s to %s.\n' "$before_version" "$after_version"
fi

printf 'Start it with: ./scripts/start-mlx-vllm.sh\n'
