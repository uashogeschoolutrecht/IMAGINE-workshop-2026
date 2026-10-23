#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/start-mlx-vlm.sh audio-file [prompt]
# Syncs the mlx-vlm dependency group into the project's shared .venv, then
# runs one audio inference.
# Use clips under 30 seconds; the notebook splits longer recordings.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# < 1 || $# > 2 )); then
    printf 'Usage: %s audio-file [prompt]\nUse a clip of at most 28 seconds, or the notebook for longer audio.\n' "$0" >&2
    exit 2
fi
AUDIO_PATH="$1"
PROMPT="${2:-Transcribe this Dutch audio verbatim into Dutch text. Output only the transcription.}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-E4B-it}"

if [[ ! -f "$AUDIO_PATH" ]]; then
    printf 'Audio file not found: %s\n' "$AUDIO_PATH" >&2
    exit 1
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    printf 'This launcher requires Apple Silicon macOS.\n' >&2
    exit 1
fi
command -v uv >/dev/null || { printf 'Install uv first.\n' >&2; exit 1; }

printf 'Model: %s\nAudio: %s\n' "$MODEL_NAME" "$AUDIO_PATH"
exec uv run --project "$PROJECT_DIR" --group mlx-vlm python -m mlx_vlm.generate \
    --model "$MODEL_NAME" \
    --audio "$AUDIO_PATH" \
    --prompt "$PROMPT" \
    --max-tokens 512 \
    --temperature 0
