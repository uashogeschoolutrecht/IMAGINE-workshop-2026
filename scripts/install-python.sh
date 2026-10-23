#!/usr/bin/env bash
set -Eeuo pipefail

# Install uv and a prebuilt Python. Do not change the system Python or pyenv.
readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_VERSION="$(cat "$PROJECT_DIR/.python-version")"

case "$(uname -s)" in
    Darwin|Linux) ;;
    *) printf 'This workshop supports macOS and Linux.\n' >&2; exit 1 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || {
        printf 'Install curl with your operating system package manager first.\n' >&2
        exit 1
    }
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || {
    printf 'uv is not on PATH. Open a new terminal and rerun this script.\n' >&2
    exit 1
}
uv python install "$PYTHON_VERSION"
printf '\nInstalled: %s\n' "$(uv --version)"
printf 'Python: %s\n' "$(uv python find --managed-python "$PYTHON_VERSION")"
printf '\nOpen a new terminal if uv was just installed, then run from the repository root:\n'
printf '  uv sync --locked\n'
printf '  uv run --no-sync jupyter lab\n'
