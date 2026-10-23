#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/start-mlx-audio.sh [audio-file] [prompt]
# Syncs the mlx-audio dependency group into the project's shared .venv. The
# first run also downloads the model from Hugging Face.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIO_PATH="${1:-$PROJECT_DIR/data/Zie-het-verschil-GrandOptical_1080p.mp3}"
PROMPT="${2:-Describe the speech and other sounds in this recording.}"
MODEL_NAME="${MODEL_NAME:-mlx-community/Qwen2-Audio-7B-Instruct-4bit}"

if [[ ! -f "$AUDIO_PATH" ]]; then
    printf 'Audio file not found: %s\n' "$AUDIO_PATH" >&2
    exit 1
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    printf 'This launcher requires Apple Silicon macOS.\n' >&2
    exit 1
fi
command -v uv >/dev/null || { printf 'Install uv first.\n' >&2; exit 1; }

printf 'Model: %s\nAudio: %s\nPrompt: %s\n' "$MODEL_NAME" "$AUDIO_PATH" "$PROMPT"
exec uv run --project "$PROJECT_DIR" --group mlx-audio python -m mlx_audio.stt.generate \
    --model "$MODEL_NAME" \
    --audio "$AUDIO_PATH" \
    --prompt "$PROMPT" \
    --max-tokens 256 \
    --output-path - \
    --format txt
