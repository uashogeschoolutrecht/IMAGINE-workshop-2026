import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

import livecodebench as lcb


def test_endpoint_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:11434/v1/")
    monkeypatch.setenv("OLLAMA_MODEL", "other")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    assert lcb.endpoint("ollama") == {
        "base_url": "http://example:11434",
        "model": "other",
        "api_key": "secret",
    }
    assert lcb.headers(lcb.endpoint("ollama")) == {"Authorization": "Bearer secret"}


def test_deterministic_selection():
    dataset = {
        "b": [{"problem_id": "2"}, {"problem_id": "1"}],
        "a": [{"problem_id": "1"}, {"problem_id": "3"}],
    }
    assert [p["problem_id"] for p in lcb.select_problems(dataset, 2)] == ["1", "2"]
    assert len(lcb.select_problems(dataset, None)) == 3


class Interface:
    prompt = "upstream prompt: "

    def generate_solution(self, statement):
        return self.call_llm(self.prompt + statement)


def evaluate(monkeypatch, response=None, error=None, code="code", verdict="Accepted"):
    mock = Mock(side_effect=error, return_value=response)
    monkeypatch.setattr(lcb, "generate", mock)
    judge = Mock(CPP_LANGUAGE="cpp")
    judge.get_result.return_value = verdict
    row = lcb.evaluate(
        {
            "problem_id": "1",
            "problem_title": "Title",
            "difficulty": "easy",
            "platform": "CF",
            "problem_statement": "statement",
        },
        {},
        SimpleNamespace(LLMInterface=Interface),
        SimpleNamespace(extract_longest_cpp_code=lambda _: code),
        judge,
        SimpleNamespace(
            max_tokens=4096, temperature=0, request_timeout=300, judge_timeout=0.01
        ),
    )
    return row, mock, judge


def test_response_and_prompt(monkeypatch):
    row, generate, judge = evaluate(
        monkeypatch,
        {
            "text_response": "cpp",
            "finish_reason": "length",
            "latency_s": 2,
            "usage": {},
        },
    )
    assert row["judge_result"] == "Accepted"
    assert row["finish_reason"] == "length"
    assert generate.call_args.args[1] == "upstream prompt: statement"
    judge.submit.assert_called_once_with("1", "cpp", "code")


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("secret"),
        ValueError("bad response"),
        KeyError("choices"),
        IndexError(),
    ],
)
def test_model_failure(monkeypatch, error):
    row, _, judge = evaluate(monkeypatch, error=error)
    assert row["judge_result"] == "Model Failed"
    assert row["error"] == type(error).__name__
    assert "secret" not in json.dumps(row)
    judge.submit.assert_not_called()


def test_no_code(monkeypatch):
    row, _, judge = evaluate(monkeypatch, {"text_response": "explanation"}, code=None)
    assert row["judge_result"] == "No Code"
    judge.submit.assert_not_called()


def test_judge_timeout(monkeypatch):
    row, _, _ = evaluate(monkeypatch, {"text_response": "cpp"}, verdict="Judging")
    assert row["judge_result"] == "Judge Timeout"


def test_judge_failure(monkeypatch):
    monkeypatch.setattr(
        lcb, "wait_verdict", Mock(side_effect=requests.ConnectionError())
    )
    row, _, _ = evaluate(monkeypatch, {"text_response": "cpp"})
    assert row["judge_result"] == "Judge Failed"


@pytest.mark.parametrize(
    "data", [{}, {"choices": []}, {"choices": [{"message": {"content": None}}]}]
)
def test_malformed_response(monkeypatch, data):
    response = Mock()
    response.json.return_value = data
    monkeypatch.setattr(lcb.requests, "post", Mock(return_value=response))
    with pytest.raises((ValueError, TypeError, KeyError, IndexError)):
        lcb.generate(
            {"base_url": "http://local", "model": "x", "api_key": ""},
            "prompt",
            4096,
            0,
            3,
        )


def run(backend="vllm"):
    return {
        "schema_version": 1,
        "backend": backend,
        "model": "model_1 & test",
        "dataset": lcb.DATASET,
        "dataset_revision": "abc",
        "upstream_commit": lcb.UPSTREAM_COMMIT,
        "selection": "subset",
        "problem_ids": ["1", "2", "3"],
        "settings": {"temperature": 0, "max_tokens": 4096},
        "results": [
            {
                "problem_id": "1",
                "judge_result": "Accepted",
                "latency_s": 2,
                "finish_reason": "stop",
            },
            {
                "problem_id": "2",
                "judge_result": "No Code",
                "latency_s": 4,
                "finish_reason": "length",
            },
        ],
    }


def test_summary_incomplete():
    assert lcb.summary(run()) == ["1/3", "33.3\\%", "1/3", "1", "3.00 s", "1"]


def test_save_and_render(tmp_path):
    lcb.save(tmp_path / "vllm.json", run())
    output = tmp_path / "slides.tex"
    lcb.render(tmp_path, output)
    text = output.read_text()
    assert "model\\_1 \\& test" in text
    assert "1/3" in text and "Not run" in text
    assert "Not evaluated & 1 & Not run & Not run" in text
    assert not (tmp_path / "vllm.tmp").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem_ids", ["9"]),
        ("settings", {"max_tokens": 800}),
        ("dataset_revision", "different"),
    ],
)
def test_mismatched_runs(tmp_path, field, value):
    lcb.save(tmp_path / "vllm.json", run())
    other = run("ollama")
    other[field] = value
    lcb.save(tmp_path / "ollama.json", other)
    with pytest.raises(ValueError, match="mismatched"):
        lcb.load_runs(tmp_path)


def test_duplicate_problem_rejected(tmp_path):
    data = run()
    data["results"].append(data["results"][0])
    lcb.save(tmp_path / "vllm.json", data)
    with pytest.raises(ValueError, match="duplicate"):
        lcb.load_runs(tmp_path)


def test_empty_render(tmp_path):
    output = tmp_path / "slides.tex"
    lcb.render(tmp_path, output)
    assert output.read_text().count("Not run") == 21


def test_escape():
    assert (
        lcb.latex("_&%$#{}~^\\")
        == r"\_\&\%\$\#\{\}\textasciitilde{}\textasciicircum{}\textbackslash{}"
    )


def test_preflight_stops_before_inference(monkeypatch):
    monkeypatch.setattr(
        lcb.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=1, stdout="")),
    )
    get = Mock()
    monkeypatch.setattr(lcb.requests, "get", get)
    with pytest.raises(RuntimeError, match="Start Docker Desktop"):
        lcb.preflight(["vllm"])
    get.assert_not_called()


def test_preflight_model_discovery(monkeypatch):
    monkeypatch.setattr(
        lcb.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=0, stdout="linux\n")),
    )
    response = Mock()
    response.json.return_value = {"data": [{"id": "wrong"}]}
    monkeypatch.setattr(lcb.requests, "get", Mock(return_value=response))
    with pytest.raises(ValueError, match="not advertised"):
        lcb.preflight(["vllm"])


def test_checkpoint_survives_interruption(monkeypatch, tmp_path):
    monkeypatch.setattr(lcb, "preflight", Mock())
    monkeypatch.setattr(lcb, "load_upstream", Mock(return_value=(None, None, None)))
    context = Mock()
    context.__enter__ = Mock(return_value=Mock())
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(lcb, "make_judge", Mock(return_value=context))
    monkeypatch.setattr(lcb, "smoke", Mock())
    monkeypatch.setitem(
        lcb.sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=lambda *a, **kw: {
                "test": [{"problem_id": "1"}, {"problem_id": "2"}]
            }
        ),
    )
    monkeypatch.setitem(
        lcb.sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Mock())
    )
    monkeypatch.setenv("VLLM_API_KEY", "private-key")
    monkeypatch.setattr(
        lcb,
        "evaluate",
        Mock(
            side_effect=[
                {"problem_id": "1", "judge_result": "Accepted"},
                KeyboardInterrupt(),
            ]
        ),
    )
    args = SimpleNamespace(
        models=["vllm"],
        upstream=tmp_path,
        workers=2,
        judge_timeout=300,
        command="run",
        dataset_revision="pinned",
        full=False,
        limit=10,
        max_tokens=4096,
        temperature=0,
        request_timeout=300,
        output=tmp_path / "results",
    )
    with pytest.raises(KeyboardInterrupt):
        lcb.execute(args)
    data = json.loads((args.output / "vllm.json").read_text())
    assert data["problem_ids"] == ["1", "2"]
    assert len(data["results"]) == 1
    assert "private-key" not in json.dumps(data)
    context.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "url", ["http://user:secret@host", "http://host?key=secret", "file:///tmp/model"]
)
def test_credentials_in_url_rejected(monkeypatch, url):
    monkeypatch.setenv("VLLM_BASE_URL", url)
    with pytest.raises(ValueError):
        lcb.endpoint("vllm")


def test_existing_results_rejected_before_docker(monkeypatch, tmp_path):
    lcb.save(tmp_path / "vllm.json", run())
    preflight = Mock()
    monkeypatch.setattr(lcb, "preflight", preflight)
    with pytest.raises(ValueError, match="already contains"):
        lcb.execute(SimpleNamespace(command="run", models=["vllm"], output=tmp_path))
    preflight.assert_not_called()


def test_discovery_failure_names_backend_without_credentials(monkeypatch):
    monkeypatch.setattr(
        lcb.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=0, stdout="linux\n")),
    )
    monkeypatch.setattr(
        lcb.requests, "get", Mock(side_effect=requests.ConnectionError("private-key"))
    )
    with pytest.raises(RuntimeError, match="spark: model discovery failed") as error:
        lcb.preflight(["spark"])
    assert "private-key" not in str(error.value)


def test_judge_reuses_image_and_removes_only_its_container(monkeypatch, tmp_path):
    docker = SimpleNamespace(
        from_env=Mock(),
        errors=SimpleNamespace(ImageNotFound=type("ImageNotFound", (Exception,), {})),
    )
    monkeypatch.setitem(lcb.sys.modules, "docker", docker)

    client = Mock()
    container = client.containers.run.return_value
    container.ports = {"8081/tcp": [{"HostPort": "12345"}]}
    container.status = "running"
    monkeypatch.setattr(docker, "from_env", Mock(return_value=client))
    monkeypatch.setattr(lcb.requests, "get", Mock(return_value=Mock()))
    (tmp_path / "LightCPVerifier").mkdir()
    module = SimpleNamespace(
        LightCPVerifierJudge=MockJudgeBase, SupportedLanguage=SimpleNamespace(CPP="cpp")
    )
    with lcb.make_judge(module, tmp_path, 2, 10) as judge:
        assert judge.base_url == "http://localhost:12345"
    client.api.build.assert_not_called()
    assert client.containers.run.call_args.kwargs["name"].startswith(
        "workshop-livecodebench-"
    )
    assert client.containers.run.call_args.kwargs["platform"] == "linux/amd64"
    assert client.containers.run.call_args.kwargs["cgroupns"] == "host"
    client.containers.get.assert_not_called()
    container.remove.assert_called_once_with(force=True)
    client.close.assert_called_once()


class MockJudgeBase:
    def __init__(self, worker):
        self.worker = worker


def test_judge_build_error_is_actionable_and_closes_client(monkeypatch, tmp_path):
    docker = SimpleNamespace(
        from_env=Mock(),
        errors=SimpleNamespace(ImageNotFound=type("ImageNotFound", (Exception,), {})),
    )
    monkeypatch.setitem(lcb.sys.modules, "docker", docker)

    client = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    client.api.build.return_value = [{"error": "build step failed"}]
    (tmp_path / "LightCPVerifier").mkdir()
    (tmp_path / "LightCPVerifier" / "Dockerfile").write_text(
        "RUN npm install -g npm@latest\nRUN set -eux; fake-download\n\n"
    )
    monkeypatch.setattr(docker, "from_env", Mock(return_value=client))
    module = SimpleNamespace(
        LightCPVerifierJudge=MockJudgeBase, SupportedLanguage=SimpleNamespace(CPP="cpp")
    )
    with (
        pytest.raises(
            RuntimeError, match="Judge image build failed: build step failed"
        ),
        lcb.make_judge(module, tmp_path, 2, 10),
    ):
        pass
    client.containers.run.assert_not_called()
    client.close.assert_called_once()


@pytest.mark.parametrize(
    "machine,expected",
    [("arm64", "linux/arm64"), ("aarch64", "linux/arm64"), ("x86_64", "linux/amd64")],
)
def test_native_architecture(monkeypatch, machine, expected):
    monkeypatch.setattr(lcb.platform, "machine", lambda: machine)
    assert lcb.native_judge_platform() == expected


def test_arm64_build_uses_native_binary_and_cpp_compiler():
    source = "\n".join(
        [
            "FROM --platform=linux/amd64 ubuntu:22.04",
            "RUN apt-get install -y " + chr(92),
            "    build-essential " + chr(92),
            "    rustc " + chr(92),
            "    curl",
            "RUN npm install -g npm@latest",
            "RUN set -eux; fake-download",
            "",
            "COPY src/ ./src/",
        ]
    )
    native = lcb.judge_build_source(source, "linux/arm64")
    assert "FROM --platform=linux/arm64" in native
    assert "linux_amd64" not in native
    assert "go-judge_1.13.0_linux_arm64" in native
    assert lcb.JUDGE_HASHES["arm64"] in native
    assert "build-essential" in native and "rustc" not in native
    assert "no-seccomp" not in native


def test_mixed_judge_architectures_rejected(tmp_path):
    first, second = run(), run("ollama")
    first["judge_platform"] = "linux/arm64"
    second["judge_platform"] = "linux/amd64"
    lcb.save(tmp_path / "vllm.json", first)
    lcb.save(tmp_path / "ollama.json", second)
    with pytest.raises(ValueError, match="judge architecture"):
        lcb.load_runs(tmp_path)


def test_arm64_policy_adds_only_file_stat_alias():
    policy = (lcb.ROOT / "benchmark-config" / "seccomp-arm64.yaml").read_text()
    assert "default_action: errno" in policy
    assert (
        "  aarch64:\n    syscalls:\n      - names: [fstatat]\n        action: allow"
        in policy
    )
    assert policy.count("names: [fstatat]") == 1


def test_arm64_launch_mounts_filter_readonly(monkeypatch, tmp_path):
    docker = SimpleNamespace(
        from_env=Mock(), errors=SimpleNamespace(ImageNotFound=Exception)
    )
    monkeypatch.setitem(lcb.sys.modules, "docker", docker)
    client = Mock()
    container = client.containers.run.return_value
    container.ports = {"8081/tcp": [{"HostPort": "12345"}]}
    container.status = "running"
    docker.from_env.return_value = client
    monkeypatch.setattr(lcb.requests, "get", Mock(return_value=Mock()))
    (tmp_path / "LightCPVerifier").mkdir()
    module = SimpleNamespace(
        LightCPVerifierJudge=MockJudgeBase, SupportedLanguage=SimpleNamespace(CPP="cpp")
    )
    with lcb.make_judge(module, tmp_path, 2, 10, "linux/arm64"):
        pass
    kwargs = client.containers.run.call_args.kwargs
    assert kwargs["platform"] == "linux/arm64"
    assert "-arm64" in client.containers.run.call_args.args[0]
    assert (
        kwargs["volumes"][str(lcb.ROOT / "benchmark-config" / "seccomp-arm64.yaml")][
            "mode"
        ]
        == "ro"
    )
    assert len(kwargs["command"]) == 1
    assert "-seccomp-conf /app/workshop-seccomp.yaml" in kwargs["command"][0]
    assert "no-seccomp" not in kwargs["command"][0]


def test_gated_dataset_error_is_actionable_and_redacted(monkeypatch, tmp_path):
    monkeypatch.setattr(lcb, "preflight", Mock())
    monkeypatch.setattr(lcb, "load_upstream", Mock(return_value=(None, None, None)))
    context = Mock()
    context.__enter__ = Mock(return_value=Mock())
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(lcb, "make_judge", Mock(return_value=context))
    monkeypatch.setattr(lcb, "smoke", Mock())
    monkeypatch.setitem(
        lcb.sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=Mock(side_effect=FileNotFoundError("private-token"))
        ),
    )
    monkeypatch.setitem(
        lcb.sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Mock())
    )
    args = SimpleNamespace(
        models=["ollama"],
        upstream=tmp_path,
        workers=2,
        judge_timeout=300,
        command="run",
        dataset_revision="pinned",
        output=tmp_path / "results",
    )
    with pytest.raises(RuntimeError, match="hf auth login") as error:
        lcb.execute(args)
    assert "private-token" not in str(error.value)
    assert not args.output.exists()
    context.__exit__.assert_called_once()
