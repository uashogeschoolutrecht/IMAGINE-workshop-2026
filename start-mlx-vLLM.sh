#!/usr/bin/env bash
MODEL_NAME=${1:-"mlx-community/gemma-4-e4b-it-4bit"}
~/.venv-vllm-metal/bin/vllm serve $MODEL_NAME