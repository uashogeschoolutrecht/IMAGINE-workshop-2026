#!/usr/bin/env bash
set -euo pipefail

# Usage: ./start-mlx-vlm.sh [audio-file] [prompt]
# Syncs the mlx-vlm dependency group into the project's shared .venv, then
# runs one audio inference.
# Use clips under 30 seconds; the notebook splits longer recordings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_PATH="${1:-$SCRIPT_DIR/data/Zie-het-verschil-GrandOptical_1080p.mp3}"
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
exec uv run --project "$SCRIPT_DIR" --group mlx-vlm python -m mlx_vlm.generate \
    --model "$MODEL_NAME" \
    --audio "$AUDIO_PATH" \
    --prompt "$PROMPT" \
    --max-tokens 512 \
    --temperature 0
