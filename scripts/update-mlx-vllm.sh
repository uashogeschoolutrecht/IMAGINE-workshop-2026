#!/usr/bin/env bash
set -Eeuo pipefail

curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
printf 'Updated vllm-metal. Activate it with: source %s/.venv-vllm-metal/bin/activate\n' \
    "$HOME"
