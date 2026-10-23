"""Exercise notebook inference boundaries without loading models or using a GPU."""

import base64
import json
import platform
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import nbformat
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
GEMMA = json.loads((ROOT / "notebooks/local-vlm-gemma.ipynb").read_text())


def run_cell(namespace, index):
    exec(  # noqa: S102 - execute our checked-in notebook
        compile(
            "".join(GEMMA["cells"][index]["source"]), f"gemma-cell-{index}", "exec"
        ),
        namespace,
    )


def response(data):
    result = Mock()
    result.json.return_value = data
    return result


@pytest.fixture
def gemma(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").touch()
    name = "UMC Utrecht - Jeroen Hendrikse - radiologieverslag"
    data = tmp_path / "data" / name
    data.mkdir(parents=True)
    for suffix in (".mp3", " - curated.txt", " - curated.srt"):
        (data / f"{name}{suffix}").touch()
    monkeypatch.setattr(shutil, "which", lambda name: "/mock/ffmpeg")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    for key in ("GEMMA_BACKEND", "GEMMA_MODEL", "VLLM_BASE_URL", "VLLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # Fail immediately if the Linux path tries to import any MLX package.
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    namespace = {}
    run_cell(namespace, 1)
    discovery = Mock(return_value=response({"data": [{"id": namespace["MODEL_NAME"]}]}))
    monkeypatch.setattr(requests, "get", discovery)
    return namespace


def test_linux_setup_and_discovery_without_mlx(gemma, monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8001/v1/")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    run_cell(gemma, 1)
    run_cell(gemma, 2)
    assert gemma["BACKEND"] == "vllm"
    requests.get.assert_called_once_with(
        "http://localhost:8001/v1/models",
        headers={"Authorization": "Bearer test-key"},
        timeout=10,
    )


def test_backend_selection(gemma, monkeypatch):
    select = gemma["select_gemma_backend"]
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        select("mlx")
    with pytest.raises(ValueError, match="GEMMA_BACKEND"):
        select("cuda")
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert select() == "mlx"
    assert select("vllm") == "vllm"  # A Mac may call a remote CUDA server.


def test_discovery_rejects_wrong_model(gemma):
    requests.get.return_value.json.return_value = {"data": [{"id": "another-model"}]}
    with pytest.raises(RuntimeError, match="not served"):
        run_cell(gemma, 2)


def test_discovery_http_error(gemma):
    requests.get.return_value.raise_for_status.side_effect = requests.HTTPError("401")
    with pytest.raises(requests.HTTPError):
        run_cell(gemma, 2)


def test_cuda_chunk_payload_and_transcription(gemma, monkeypatch):
    run_cell(gemma, 2)
    post = Mock(
        return_value=response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "  Hallo wereld.  "},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(requests, "post", post)
    audio = b"wav-chunk-bytes"
    assert gemma["transcribe_gemma_chunk"](audio) == "Hallo wereld."
    args, kwargs = post.call_args
    assert args == ("http://localhost:8000/v1/chat/completions",)
    payload = kwargs["json"]
    assert payload["model"] == "google/gemma-4-E4B-it"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 1024
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    content = payload["messages"][0]["content"]
    assert content[0]["text"] == gemma["TRANSCRIPTION_PROMPT"]
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "wav"
    assert base64.b64decode(content[1]["input_audio"]["data"]) == audio
    assert kwargs["timeout"] == (10, 600)


@pytest.mark.parametrize(
    "completion, message",
    [
        ({"choices": []}, "no completion choices"),
        (
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "partial"}}
                ]
            },
            "incomplete",
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]},
            "no transcription",
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
            "no transcription",
        ),
    ],
)
def test_cuda_rejects_incomplete_or_empty_transcription(
    gemma, monkeypatch, completion, message
):
    run_cell(gemma, 2)
    monkeypatch.setattr(requests, "post", Mock(return_value=response(completion)))
    with pytest.raises(RuntimeError, match=message):
        gemma["transcribe_gemma_chunk"](b"audio")


def test_cuda_http_error(gemma, monkeypatch):
    run_cell(gemma, 2)
    result = response({})
    result.raise_for_status.side_effect = requests.HTTPError("503")
    monkeypatch.setattr(requests, "post", Mock(return_value=result))
    with pytest.raises(requests.HTTPError):
        gemma["transcribe_gemma_chunk"](b"audio")


@pytest.mark.parametrize(
    "tokens,text,error",
    [
        (12, "  Hallo wereld.  ", None),
        (1024, "partial", "MAX_TOKENS"),
        (12, "", "no transcription"),
    ],
)
def test_mlx_load_and_chunk(gemma, monkeypatch, tokens, text, error):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setenv("GEMMA_BACKEND", "mlx")
    run_cell(gemma, 1)
    mlx, core, vlm, prompt_utils = (
        ModuleType(n)
        for n in (
            "mlx",
            "mlx.core",
            "mlx_vlm",
            "mlx_vlm.prompt_utils",
        )
    )
    mlx.core = core
    core.clear_cache = Mock()
    model = SimpleNamespace(config={})
    vlm.load = Mock(return_value=(model, "processor"))
    prompt_utils.apply_chat_template = Mock(return_value="formatted-prompt")

    def generate(**kwargs):
        assert Path(kwargs["audio"][0]).read_bytes() == b"audio"
        assert kwargs["prompt"] == "formatted-prompt"
        assert kwargs["temperature"] == 0
        return SimpleNamespace(text=text, generation_tokens=tokens)

    vlm.generate = Mock(side_effect=generate)
    for module in (mlx, core, vlm, prompt_utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    run_cell(gemma, 2)
    requests.get.assert_not_called()
    vlm.load.assert_called_once_with("google/gemma-4-E4B-it")
    assert prompt_utils.apply_chat_template.call_args.kwargs[
        "chat_template_kwargs"
    ] == {
        "enable_thinking": False,
    }
    if error:
        with pytest.raises(RuntimeError, match=error):
            gemma["transcribe_gemma_chunk"](b"audio")
    else:
        assert gemma["transcribe_gemma_chunk"](b"audio") == "Hallo wereld."
        core.clear_cache.assert_called_once()


@pytest.mark.parametrize("directory", [ROOT, ROOT / "notebooks"])
def test_vllm_text_run_needs_only_configured_server(directory, monkeypatch):
    monkeypatch.chdir(directory)
    monkeypatch.setenv("VLLM_BASE_URL", "http://local-test:8000/v1/")
    monkeypatch.setenv("VLLM_MODEL", "test-model")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    get = Mock(return_value=response({"data": [{"id": "test-model"}]}))
    post = Mock(
        return_value=response(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "Hello."}}
                ],
                "usage": {"completion_tokens": 3, "prompt_tokens": 5},
            }
        )
    )
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    notebook = json.loads((ROOT / "notebooks/local-llm-vllm.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "vllm-notebook", "exec"), namespace)  # noqa: S102 - execute our checked-in notebook
    assert post.call_count == 6  # One warm-up and the five workshop questions.
    assert all(
        call.args[0] == "http://local-test:8000/v1/chat/completions"
        for call in post.call_args_list
    )
    assert len(namespace["df_results"]) == 5


@pytest.mark.parametrize(
    "path", sorted((ROOT / "notebooks").glob("*.ipynb")), ids=lambda p: p.name
)
def test_notebook_format_and_python_syntax(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:{index}", "exec")
