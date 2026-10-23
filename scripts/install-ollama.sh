#!/usr/bin/env bash
set -Eeuo pipefail

case "$(uname -s)" in
    Linux)
        curl -fsSL https://ollama.com/install.sh | sh
        ;;
    Darwin)
        printf 'Install the macOS app from https://ollama.com/download/mac\n'
        printf 'Move it to Applications, open it, and follow its CLI setup prompt.\n'
        printf 'Then run: ./scripts/start-ollama.sh\n'
        ;;
    *) printf 'This workshop supports macOS and Linux.\n' >&2; exit 1 ;;
esac
