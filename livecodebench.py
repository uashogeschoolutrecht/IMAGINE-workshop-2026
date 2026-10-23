"""LiveCodeBench-Pro workshop runner and Beamer result exporter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "d614eef4f95b3654bf0758c64e1194b2575850b8"
DATASET = "QAQAQAQAQ/LiveCodeBench-Pro"
DEFAULTS = {
    "vllm": ("http://localhost:8000", "mlx-community/gemma-4-e4b-it-4bit"),
    "ollama": ("http://localhost:11434", "gemma4:26b"),
    "spark": ("http://spark-0240:9000", "deepseek-ai/DeepSeek-V4-Flash"),
}
LABELS = {"vllm": "Local vLLM", "ollama": "Ollama", "spark": "Spark"}
FAILURES = {"Model Failed", "No Code", "Judge Failed", "Judge Timeout"}


def endpoint(name):
    url, model = DEFAULTS[name]
    prefix = name.upper()
    config = {
        "base_url": os.getenv(f"{prefix}_BASE_URL", url)
        .rstrip("/")
        .removesuffix("/v1"),
        "model": os.getenv(f"{prefix}_MODEL", model),
        "api_key": os.getenv(f"{prefix}_API_KEY", ""),
    }
    parsed = urlsplit(config["base_url"])
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{prefix}_BASE_URL must be an HTTP(S) server URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "Use API_KEY for credentials; BASE_URL must not contain credentials or query strings"
        )
    return config


def headers(config):
    return {"Authorization": f"Bearer {config['api_key']}"} if config["api_key"] else {}


def judge_error_verdict(error):
    # Upstream wraps solution compiler diagnostics in its generic error status.
    # Checker compilation and sandbox failures remain infrastructure failures.
    if (
        isinstance(error, str)
        and error.startswith("Error: compile failed: main.cpp:")
        and (": error:" in error or ": fatal error:" in error)
    ):
        return "Compilation Error"
    return "Judge Failed"


def select_problems(dataset, limit):
    unique = {}
    for split in dataset.values():
        for row in split:
            unique.setdefault(str(row["problem_id"]), dict(row))
    rows = [unique[key] for key in sorted(unique)]
    return rows if limit is None else rows[:limit]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_upstream(path):
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError(
            f"Upstream checkout must be at {UPSTREAM_COMMIT}; found {actual}"
        )
    if subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(
            "Use a clean upstream checkout, including its LightCPVerifier submodule"
        )
    if not (path / "LightCPVerifier" / "Dockerfile").exists():
        raise ValueError(
            "Initialize upstream submodules with git submodule update --init --recursive"
        )
    modules = []
    for name in ("api_interface", "util", "judge"):
        spec = importlib.util.spec_from_file_location(
            f"lcb_{name}", path / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


JUDGE_HASHES = {
    "amd64": "700d0ba6a6c02cce777e8761a5851b3d8d3e198e120887cda8e9134eb62efbce",
    "arm64": "ce76a5d295e8fc65d0f621c465839e5f56016b402c593a7b2942eaa6f6da435a",
}


def native_judge_platform():
    return (
        "linux/arm64"
        if platform.machine().lower() in {"arm64", "aarch64"}
        else "linux/amd64"
    )


def judge_build_source(source, judge_platform):
    arch = judge_platform.split("/")[1]
    source = source.replace(
        "FROM --platform=linux/amd64", f"FROM --platform={judge_platform}"
    )
    upgrade = "RUN npm install -g npm@latest"
    if source.count(upgrade) != 1:
        raise RuntimeError("Unexpected upstream npm upgrade step")
    source = source.replace(upgrade, "RUN npm --version")
    # This runner generates C++ only; omit unrelated language toolchains.
    if arch == "arm64":
        omitted = {
            "python3",
            "python3-pip",
            "pypy3",
            "openjdk-17-jdk",
            "kotlin",
            "rustc",
            "cargo",
            "golang",
        }
        source = "".join(
            line
            for line in source.splitlines(keepends=True)
            if line.strip().rstrip(chr(92)).strip() not in omitted
        )
    start = source.index("RUN set -eux;")
    end = source.index("\n\n", start)
    return (
        source[:start]
        + (
            "RUN curl --retry 3 --max-time 180 -fsSL "
            "https://github.com/criyle/go-judge/releases/download/v1.13.0/"
            f"go-judge_1.13.0_linux_{arch} -o /usr/local/bin/go-judge "
            f"&& echo '{JUDGE_HASHES[arch]}  /usr/local/bin/go-judge' | sha256sum -c - "
            "&& chmod +x /usr/local/bin/go-judge"
        )
        + source[end:]
    )


def make_judge(module, path, workers, timeout, judge_platform="linux/amd64"):
    """Reuse upstream download/submit protocol, isolate Docker lifecycle and waits."""

    class WorkshopJudge(module.LightCPVerifierJudge):
        REPO_DIR = str((path / "LightCPVerifier").resolve())
        PROBLEMS_DIR = str((path / "LightCPVerifier" / "problems").resolve())
        CPP_LANGUAGE = module.SupportedLanguage.CPP
        IMAGE_NAME = (
            "workshop-livecodebench-pro:"
            + UPSTREAM_COMMIT[:12]
            + "-node20-go113-"
            + judge_platform.split("/")[1]
        )
        CONTAINER_NAME = "workshop-livecodebench-" + uuid.uuid4().hex[:12]

        def __enter__(self):
            import docker

            self.docker_client = docker.from_env(timeout=timeout)
            self.container = None
            try:
                self.docker_client.ping()
                try:
                    self.docker_client.images.get(self.IMAGE_NAME)
                    print("Using cached judge image", flush=True)
                except docker.errors.ImageNotFound:
                    print(f"Building judge image ({judge_platform})", flush=True)
                    source = judge_build_source(
                        (Path(self.REPO_DIR) / "Dockerfile").read_text(), judge_platform
                    )
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".Dockerfile"
                    ) as buildfile:
                        buildfile.write(source)
                        buildfile.flush()
                        build_tail = []
                        for event in self.docker_client.api.build(
                            path=self.REPO_DIR,
                            tag=self.IMAGE_NAME,
                            dockerfile=buildfile.name,
                            platform=judge_platform,
                            decode=True,
                            rm=True,
                        ):
                            if event.get("error"):
                                detail = "\n".join(build_tail[-12:])
                                raise RuntimeError(
                                    f"Judge image build failed: {event['error']}\n{detail}"
                                )
                            message = event.get("stream", "").strip()
                            if message:
                                build_tail.append(message)
                            if message.startswith("Step "):
                                print(message, flush=True)
                    self.docker_client.images.get(self.IMAGE_NAME)
                for folder in ("problems", "submissions", "data"):
                    (Path(self.REPO_DIR) / folder).mkdir(exist_ok=True)
                volumes = {
                    str(Path(self.REPO_DIR) / folder): {
                        "bind": f"/app/{folder}",
                        "mode": "rw",
                    }
                    for folder in ("problems", "submissions", "data")
                }
                launch = {}
                if judge_platform == "linux/arm64":
                    volumes[str(ROOT / "benchmark-config" / "seccomp-arm64.yaml")] = {
                        "bind": "/app/workshop-seccomp.yaml",
                        "mode": "ro",
                    }
                    launch = {
                        "entrypoint": ["/bin/sh", "-c"],
                        "command": [
                            (
                                '/usr/local/bin/go-judge -parallelism "$JUDGE_WORKERS" '
                                "-seccomp-conf /app/workshop-seccomp.yaml & sandbox_pid=$!; "
                                "for i in $(seq 1 30); do "
                                "curl -fsS http://localhost:5050/version >/dev/null 2>&1 "
                                "&& exec node /app/server.js; "
                                'kill -0 "$sandbox_pid" 2>/dev/null || exit 1; sleep 1; done; exit 1'
                            )
                        ],
                    }
                self.container = self.docker_client.containers.run(
                    self.IMAGE_NAME,
                    name=self.CONTAINER_NAME,
                    platform=judge_platform,
                    privileged=True,
                    cgroupns="host",
                    detach=True,
                    shm_size="4g",
                    environment={
                        "JUDGE_WORKERS": str(workers),
                        "GJ_PARALLELISM": str(workers),
                    },
                    volumes=volumes,
                    **launch,
                    ports={"8081/tcp": ("127.0.0.1", None)},
                )
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    self.container.reload()
                    ports = self.container.ports.get("8081/tcp")
                    if ports:
                        self.base_url = f"http://localhost:{ports[0]['HostPort']}"
                        try:
                            requests.get(
                                self.base_url + "/health", timeout=2
                            ).raise_for_status()
                            return self
                        except requests.RequestException:
                            pass
                    if self.container.status == "exited":
                        raise RuntimeError(
                            f"Judge exited ({judge_platform}); check startup logs and cgroups"
                        )
                    time.sleep(1)
                raise TimeoutError("Judge startup timed out; check Docker Desktop")
            except BaseException:
                self.__exit__(None, None, None)
                raise

        def __exit__(self, *_):
            if self.container is not None:
                self.container.remove(force=True)
            self.docker_client.close()

        def submit(self, problem_id, language, code):
            self._ensure_data_downloaded(problem_id)
            response = requests.post(
                self.base_url + "/submit",
                json={
                    "pid": problem_id,
                    "lang": language.value,
                    "code": code,
                },
                timeout=(10, timeout),
            )
            response.raise_for_status()
            return response.json()["sid"]

        def get_result(self, submission_id):
            response = requests.get(
                f"{self.base_url}/result/{submission_id}", timeout=(10, timeout)
            )
            if response.status_code == 404:
                return "Judging"
            response.raise_for_status()
            result = response.json()
            if result["status"] == "queued":
                return "Judging"
            if result["status"] == "error":
                self.last_error = result.get("error", "Unknown judge error")
                return judge_error_verdict(self.last_error)
            return result["result"]

    return WorkshopJudge(worker=workers)


def wait_verdict(judge, sid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        verdict = judge.get_result(sid)
        if verdict != "Judging":
            return verdict
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    return "Judge Timeout"


# Codeforces 2000A: a valid power starts with 10 and has an exponent >= 2,
# with no leading zero. This problem is also used by upstream's judge example.
SMOKE_CODE = """#include <bits/stdc++.h>
using namespace std;
int main(){int t;cin>>t;while(t--){string s;cin>>s;
bool ok=s.size()>2 && s.substr(0,2)=="10" && s[2]!='0';
if(ok && s.size()==3) ok=s[2]>='2';
cout<<(ok?"YES":"NO")<<"\\n";}}
"""


def preflight(names):
    result = subprocess.run(
        ["docker", "info", "--format", "{{.OSType}}"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode or result.stdout.strip() != "linux":
        raise RuntimeError(
            "Docker Linux daemon unavailable. Start Docker Desktop and rerun check."
        )
    for name in names:
        config = endpoint(name)
        try:
            response = requests.get(
                config["base_url"] + "/v1/models",
                headers=headers(config),
                timeout=(5, 20),
            )
            response.raise_for_status()
            models = {row["id"] for row in response.json()["data"]}
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(
                f"{name}: model discovery failed ({type(error).__name__}); "
                f"start the server or update {name.upper()}_BASE_URL / API_KEY"
            ) from None
        if config["model"] not in models:
            raise ValueError(
                f"{name}: configured model is not advertised; update {name.upper()}_MODEL"
            )


def smoke(judge, module, timeout):
    sid = judge.submit("2000A", module.SupportedLanguage.CPP, SMOKE_CODE)
    verdict = wait_verdict(judge, sid, timeout)
    if verdict == "Accepted":
        print("Known-solution judge check passed (seccomp enabled)", flush=True)
    if verdict != "Accepted":
        logs = judge.container.logs(tail=40).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Judge compatibility check returned {verdict}; no model evaluation started\n"
            f"{getattr(judge, 'last_error', '')}\n{logs}"
        )


def generate(config, prompt, max_tokens, temperature, timeout):
    start = time.perf_counter()
    response = requests.post(
        config["base_url"] + "/v1/chat/completions",
        headers=headers(config),
        json={
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        timeout=(10, timeout),
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("Expected string message.content")
    return {
        "text_response": content,
        "reasoning_text": message.get("reasoning_content")
        or message.get("reasoning")
        or "",
        "latency_s": time.perf_counter() - start,
        "usage": data.get("usage") or {},
        "finish_reason": choice.get("finish_reason"),
        "served_model": data.get("model"),
    }


def evaluate(problem, config, interface, util, judge, args):
    row = {
        key: problem[key]
        for key in ("problem_id", "problem_title", "difficulty", "platform")
    }
    row.update(
        text_response=None,
        code=None,
        latency_s=None,
        usage={},
        finish_reason=None,
        judge_result="Model Failed",
    )
    try:

        class EndpointLLM(interface.LLMInterface):
            def call_llm(self, user_prompt):
                result = generate(
                    config,
                    user_prompt,
                    args.max_tokens,
                    args.temperature,
                    args.request_timeout,
                )
                return result["text_response"], result

        _, metadata = EndpointLLM().generate_solution(problem["problem_statement"])
        row.update(metadata)
    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        # Exception messages can contain URLs/credentials; retain only their type.
        row["error"] = type(error).__name__
        return row
    row["code"] = util.extract_longest_cpp_code(row["text_response"])
    if not row["code"]:
        row["judge_result"] = "No Code"
        return row
    try:
        sid = judge.submit(row["problem_id"], judge.CPP_LANGUAGE, row["code"])
        row["judge_result"] = wait_verdict(judge, sid, args.judge_timeout)
    except Exception as error:  # noqa: BLE001 - isolate upstream failures and redact CLI errors
        row["judge_result"] = "Judge Failed"
        row["error"] = type(error).__name__
    return row


def execute(args):
    if args.command == "run" and any(
        (args.output / f"{name}.json").exists() for name in args.models
    ):
        raise ValueError(
            "Output already contains results; choose a new --output directory"
        )
    preflight(args.models)
    interface, util, module = load_upstream(args.upstream)
    judge_platform = getattr(args, "judge_platform", "linux/amd64")
    with make_judge(
        module, args.upstream, args.workers, args.judge_timeout, judge_platform
    ) as judge:
        smoke(judge, module, args.judge_timeout)
        if args.command == "check":
            print("Docker, endpoints, and C++ judge compatibility check passed")
            return
        from datasets import load_dataset
        from huggingface_hub import HfApi

        try:
            revision = args.dataset_revision or HfApi().dataset_info(DATASET).sha
            dataset = load_dataset(DATASET, revision=revision)
        except Exception as error:  # noqa: BLE001 - redact external access errors
            raise RuntimeError(
                f"Dataset access failed ({type(error).__name__}). "
                f"Accept the conditions at https://huggingface.co/datasets/{DATASET} "
                "and authenticate locally with hf auth login or HF_TOKEN, then rerun."
            ) from None
        problems = select_problems(dataset, None if args.full else args.limit)
        if not problems:
            raise ValueError("Dataset selection is empty")
        settings = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "attempts": 1,
            "request_timeout": args.request_timeout,
            "judge_timeout": args.judge_timeout,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        if any((args.output / f"{name}.json").exists() for name in args.models):
            raise ValueError(
                "Output already contains results; choose a new --output directory"
            )
        for name in args.models:
            config = endpoint(name)
            run = {
                "schema_version": 1,
                "backend": name,
                "model": config["model"],
                "base_url": config["base_url"],
                "upstream_commit": UPSTREAM_COMMIT,
                "dataset": DATASET,
                "dataset_revision": revision,
                "problem_ids": [str(p["problem_id"]) for p in problems],
                "selection": "full" if args.full else "subset",
                "settings": settings,
                "started_at": datetime.now(UTC).isoformat(),
                "judge_platform": judge_platform,
                "judge_build": "node20-bundled-npm-go-judge-1.13.0"
                + ("-arm64-fstatat" if judge_platform == "linux/arm64" else ""),
                "judge_workers": args.workers,
                "results": [],
            }
            target = args.output / f"{name}.json"
            save(target, run)
            for problem in problems:
                row = evaluate(problem, config, interface, util, judge, args)
                run["results"].append(row)
                save(target, run)
                print(f"{name} {row['problem_id']}: {row['judge_result']}")


def latex(value):
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(c, c) for c in str(value)).replace("\n", " ")


def load_runs(directory):
    runs = {}
    reference = None
    for name in DEFAULTS:
        path = directory / f"{name}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        if run["backend"] != name or run["schema_version"] != 1:
            raise ValueError("Invalid result backend or schema version")
        signature = [
            run[key]
            for key in (
                "dataset",
                "dataset_revision",
                "upstream_commit",
                "problem_ids",
                "settings",
                "selection",
            )
        ]
        signature += [run.get("judge_platform"), run.get("judge_build")]
        if reference is not None and signature != reference:
            raise ValueError(
                "Runs have mismatched problems, dataset revision, generation settings, or judge architecture"
            )
        reference = signature
        ids = run["problem_ids"]
        recorded = [str(row["problem_id"]) for row in run["results"]]
        if (
            not ids
            or len(ids) != len(set(ids))
            or len(recorded) != len(set(recorded))
            or not set(recorded) <= set(ids)
        ):
            raise ValueError("Invalid or duplicate problem IDs")
        runs[name] = run
    return runs


def summary(run):
    rows = run["results"]
    accepted = sum(row["judge_result"] == "Accepted" for row in rows)
    completed = sum(row["judge_result"] not in FAILURES | {"Judging"} for row in rows)
    timings = [row["latency_s"] for row in rows if row.get("latency_s") is not None]
    return [
        f"{accepted}/{len(run['problem_ids'])}",
        f"{100 * accepted / len(run['problem_ids']):.1f}\\%",
        f"{completed}/{len(run['problem_ids'])}",
        str(sum(row["judge_result"] in FAILURES for row in rows)),
        f"{sum(timings) / len(timings):.2f} s" if timings else "---",
        str(sum(row.get("finish_reason") == "length" for row in rows)),
    ]


def render(directory, output):
    runs = load_runs(directory)
    lines = [
        r"\section{LiveCodeBench-Pro}",
        r"\begin{frame}{LiveCodeBench-Pro: Methodology}",
        r"\small",
    ]
    if runs:
        run = next(iter(runs.values()))
        settings = run["settings"]
        selection = "Full dataset" if run["selection"] == "full" else "Workshop subset"
        if run.get("judge_platform"):
            lines.append(f"Judge architecture: {latex(run['judge_platform'])}.\\\\")
        lines += [
            f"{selection}: {len(run['problem_ids'])} identical problems per model.\\\\",
            f"One C++ solution attempt; temperature {latex(settings['temperature'])}; token budget {latex(settings['max_tokens'])}.\\\\",
            f"Dataset revision: \\texttt{{{latex(run['dataset_revision'][:12])}}}.\\\\",
            f"Upstream commit: \\texttt{{{latex(run['upstream_commit'][:12])}}}.\\\\",
        ]
    else:
        lines += [
            r"Planned workshop subset: 10 identical problems per model.\\",
            r"One C++ solution attempt; temperature 0; token budget 4,096.\\",
            r"No saved LiveCodeBench-Pro evaluations are available.",
        ]
    lines += [
        r"\vfill",
        r"Acceptance = accepted / selected problems. Missing or failed evaluations do not count as accepted.\\",
        r"Subset scores are not full-benchmark leaderboard scores. Inference latency excludes judging.\\",
    ]
    if "spark" in runs and runs["spark"].get("base_url"):
        lines.append(
            f"Spark endpoint: \\texttt{{{latex(runs['spark']['base_url'])}}}.\\\\"
        )
    lines += [
        r"\href{https://github.com/GavinZhengOI/LiveCodeBench-Pro}{Source: LiveCodeBench-Pro}",
        r"\end{frame}",
        r"\begin{frame}{LiveCodeBench-Pro: Three-Model Results}",
        r"\small",
        r"\begin{tabular}{lccc}\toprule",
        r"Metric & Local vLLM & Ollama & Spark \\\midrule",
    ]
    values = {
        name: summary(runs[name]) if name in runs else ["Not run"] * 6
        for name in DEFAULTS
    }
    for i, label in enumerate(
        (
            "Accepted / selected",
            "Acceptance",
            "Completed judging",
            "Failures",
            "Mean inference latency",
            "Truncated responses",
        )
    ):
        lines.append(
            label + " & " + " & ".join(values[name][i] for name in DEFAULTS) + r" \\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\vfill\footnotesize",
        r"Completed judging includes compilation errors and definitive code verdicts. Failures include inference errors, missing code, judge errors and timeouts.\\",
        r"Incomplete runs remain visible in the selected-problem denominator. Truncations may also have a judge verdict.\\",
    ]
    for name, run in runs.items():
        date = (
            " ("
            + datetime.fromisoformat(run["started_at"])
            .astimezone(ZoneInfo("Europe/Amsterdam"))
            .date()
            .isoformat()
            + ")"
            if run.get("started_at")
            else ""
        )
        lines.append(latex(LABELS[name] + ": " + run["model"] + date) + r"\\")
    lines += [
        r"\end{frame}",
        r"\begin{frame}{LiveCodeBench-Pro: Verdict Breakdown}",
        r"\small",
    ]
    verdicts = sorted(
        {row["judge_result"] for run in runs.values() for row in run["results"]}
    )
    # Paginate verdict tables rather than shrink text when new verdicts appear.
    for offset in range(0, max(1, len(verdicts)), 8):
        if offset:
            lines += [
                r"\end{frame}",
                r"\begin{frame}{LiveCodeBench-Pro: Verdict Breakdown (continued)}",
                r"\small",
            ]
        lines += [
            r"\begin{tabular}{lccc}\toprule",
            r"Verdict & Local vLLM & Ollama & Spark \\\midrule",
        ]
        for verdict in verdicts[offset : offset + 8]:
            counts = [
                str(
                    sum(row["judge_result"] == verdict for row in runs[name]["results"])
                )
                if name in runs
                else "Not run"
                for name in DEFAULTS
            ]
            lines.append(latex(verdict) + " & " + " & ".join(counts) + r" \\")
        missing = [
            str(len(runs[name]["problem_ids"]) - len(runs[name]["results"]))
            if name in runs
            else "Not run"
            for name in DEFAULTS
        ]
        lines += [
            "Not evaluated & " + " & ".join(missing) + r" \\",
            r"\bottomrule\end{tabular}",
        ]
    lines += [
        r"\vfill\footnotesize Different models, quantization and reasoning behavior limit backend comparisons.",
        r"\end{frame}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Generated {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "run"):
        p = sub.add_parser(command)
        p.add_argument(
            "--upstream", type=Path, default=ROOT / ".benchmark" / "LiveCodeBench-Pro"
        )
        p.add_argument(
            "--models", nargs="+", choices=list(DEFAULTS), default=list(DEFAULTS)
        )
        p.add_argument(
            "--judge-platform",
            choices=["linux/arm64", "linux/amd64"],
            default=native_judge_platform(),
        )
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--judge-timeout", type=float, default=300)
        p.add_argument("--request-timeout", type=float, default=300)
        p.add_argument("--max-tokens", type=int, default=4096)
        p.add_argument("--temperature", type=float, default=0)
        p.add_argument("--limit", type=int, default=10)
        p.add_argument("--full", action="store_true")
        p.add_argument("--dataset-revision")
        p.add_argument("--output", type=Path, default=ROOT / "benchmark-results")
    p = sub.add_parser("render")
    p.add_argument("--results", type=Path, default=ROOT / "benchmark-results")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "presentation" / "livecodebench-results.tex",
    )
    args = parser.parse_args()
    if args.command != "render":
        args.models = list(dict.fromkeys(args.models))
    try:
        if args.command == "render":
            render(args.results, args.output)
        else:
            if (
                min(
                    args.limit,
                    args.max_tokens,
                    args.workers,
                    args.request_timeout,
                    args.judge_timeout,
                )
                <= 0
            ):
                parser.error("Limits, workers, and timeouts must be positive")
            execute(args)
    except KeyboardInterrupt:
        print("Interrupted; completed problem results remain saved.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - isolate upstream failures and redact CLI errors
        # Avoid printing network exceptions that might contain authentication data.
        print(
            str(error)
            if isinstance(error, (ValueError, RuntimeError))
            else f"{type(error).__name__}: check Docker, dependencies and endpoint configuration",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
