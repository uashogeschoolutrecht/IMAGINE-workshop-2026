"""Check launcher platform selection and argument forwarding with fake commands."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path, body):
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def commands(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(
        bin_dir / "uname",
        'if [[ "$1" == "-s" ]]; then echo "$TEST_OS"; else echo arm64; fi',
    )
    log = tmp_path / "args"
    environment = dict(
        os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", TEST_LOG=str(log)
    )
    return bin_dir, log, environment


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            ["--port", "8001"],
            ["serve", "mlx-community/gemma-4-e4b-it-4bit", "--port", "8001"],
        ),
        (
            ["openai/whisper-large-v3", "--port", "8002"],
            ["serve", "openai/whisper-large-v3", "--port", "8002"],
        ),
    ],
)
def test_metal_forwards_model_and_flags(commands, args, expected):
    bin_dir, log, environment = commands
    executable(bin_dir / "vllm", 'printf "%s\\n" "$@" > "$TEST_LOG"')
    environment.update(TEST_OS="Darwin", VLLM_BIN=str(bin_dir / "vllm"))
    subprocess.run(
        ["bash", str(ROOT / "scripts/start-mlx-vllm.sh"), *args],
        env=environment,
        check=True,
    )
    assert log.read_text().splitlines() == expected


def test_cuda_forwards_model_and_flags(commands):
    bin_dir, log, environment = commands
    executable(bin_dir / "python", "exit 0")
    executable(
        bin_dir / "uv",
        """if [[ "$*" == *"vllm serve"* ]]; then
    printf '%s\\n' "$@" > "$TEST_LOG"
else
    echo 'mock CUDA runtime'
fi""",
    )
    environment.update(TEST_OS="Linux", VLLM_VENV=str(bin_dir.parent))
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/start-cuda-vllm.sh"),
            "--port",
            "8001",
        ],
        env=environment,
        check=True,
    )
    assert log.read_text().splitlines()[-5:] == [
        "vllm",
        "serve",
        "google/gemma-4-E4B-it",
        "--port",
        "8001",
    ]


def test_ollama_mac_preloads_selected_model_without_nvidia(commands):
    bin_dir, log, environment = commands
    executable(bin_dir / "nvidia-smi", "exit 1")
    executable(bin_dir / "curl", 'printf "%s\\n" "$@" >> "$TEST_LOG"')
    executable(
        bin_dir / "ollama",
        """if [[ "$1" == "ps" ]]; then
    echo 'gemma4:e2b id 5GB 100% GPU'
fi""",
    )
    environment.update(TEST_OS="Darwin")
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/start-ollama.sh"),
            "gemma4:e2b",
        ],
        env=environment,
        check=True,
    )
    assert '"model":"gemma4:e2b"' in log.read_text()
    assert "gemma4:26b" not in log.read_text()


def test_mlx_audio_requires_an_explicit_file():
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/start-mlx-vlm.sh"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "audio-file" in result.stderr
