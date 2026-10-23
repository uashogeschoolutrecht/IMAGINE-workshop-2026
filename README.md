# Local LLM workshop

A hands-on starting point for running and testing local LLMs with Ollama and vLLM, alongside a Spark-hosted endpoint. The workshop notebooks retain the existing smoke benchmarks and transcription evaluations; the Beamer presentation compares their saved outputs.

## Python environment

The project uses one shared `.venv`. The default dependencies are installed by
`uv sync`; add either MLX use case to that same environment when needed:

```sh
uv sync
uv sync --group mlx-vlm
uv sync --group mlx-audio
# Or install all three use cases together:
uv sync --group mlx-vlm --group mlx-audio
```

`start-mlx-vlm.sh` and `start-mlx-audio.sh` select their corresponding group
automatically. The vLLM Metal launcher is separate because the upstream
vLLM Metal installer manages `~/.venv-vllm-metal` itself.

## Notebooks

All workshop notebooks live in `notebooks/`. Select the repository's shared
`.venv/bin/python` kernel and run cells in order. Notebook setup cells locate
the repository root automatically, so project imports and data paths work when
the kernel starts in either the repository root or `notebooks/`.

Pyrefly and VS Code/Pylance are configured to use the repository root as an
import search path. This keeps imports from root-level project modules such as
`livecodebench.py` resolvable from notebooks and regular Python files.

## LiveCodeBench-Pro coding evaluation

`livecodebench.py` evaluates C++ solutions using [LiveCodeBench-Pro](https://github.com/GavinZhengOI/LiveCodeBench-Pro). The default is a workshop subset of the first 10 unique problem IDs in sorted order, shared by all three models. This is not a representative sample or a full leaderboard score.

Install the optional benchmark group and initialize a clean, pinned upstream checkout:

```sh
uv sync --group dev
mkdir -p .benchmark
git clone --recurse-submodules https://github.com/GavinZhengOI/LiveCodeBench-Pro.git .benchmark/LiveCodeBench-Pro
git -C .benchmark/LiveCodeBench-Pro checkout d614eef4f95b3654bf0758c64e1194b2575850b8
git -C .benchmark/LiveCodeBench-Pro submodule update --init --recursive
```

Start Docker Desktop and the inference servers. The judge defaults to the Python host's native architecture: `linux/arm64` on Apple Silicon, `linux/amd64` on x86. Use `--judge-platform` to override this when Docker runs on a different host. Generated C++ solutions and their source checkers compile inside the judge with the native compiler. ARM64 builds omit unrelated language toolchains; this runner supports C++ evaluation only.

The privileged judge uses Docker's host cgroup namespace within the Linux VM so go-judge can create its nested resource controls. Seccomp stays enabled. The vendored ARM64 policy adds only `fstatat` (syscall 79), the native equivalent of x86 `newfstatat`, omitted by go-judge 1.13.0's generated filter. The policy is mounted read-only; the launcher waits for the sandbox service before starting the frontend. The runner uses unique container names, loopback-only ports, and cleans up only its own container. Initial image and dataset downloads can take several minutes; build steps are printed and later runs reuse architecture-specific cached images.

A temporary Dockerfile retains Node 20's bundled npm instead of upgrading to incompatible npm 12, and pins go-judge 1.13.0 for the chosen architecture with its release checksum. It also corrects the upstream filter that selected go-judge-init. The pinned upstream checkout stays unchanged.

| Backend | Default endpoint | Default model | Environment overrides |
| --- | --- | --- | --- |
| vLLM | `http://localhost:8000` | `mlx-community/gemma-4-e4b-it-4bit` | `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY` |
| Ollama | `http://localhost:11434` | `gemma4:26b` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY` |
| Spark | `http://spark-0240:8000` | `deepseek-ai/DeepSeek-V4-Flash` | `SPARK_BASE_URL`, `SPARK_MODEL`, `SPARK_API_KEY` |

Base URLs may include a trailing `/v1`; credentials belong in the API-key environment variables. Use the exact model IDs advertised by each server. Model requests originate from the host Python process, so localhost endpoints do not need Docker networking changes.

```sh
uv run python livecodebench.py check
uv run python livecodebench.py run
uv run python livecodebench.py render
cd presentation
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
```

The amd64 sandbox on this Apple Silicon Docker Desktop setup failed with `install seccomp in container init: invalid argument` under emulation. Use the native ARM64 build on this Mac, keeping seccomp enabled. If the native check also fails, it prints the sandbox startup logs and stops before inference. A native Linux host remains an alternative; its inference URLs must point to reachable servers rather than this Mac's localhost.

The native ARM64 build passed the known correct Codeforces 2000A solution check on 8 October 2026 with seccomp enabled. The original amd64 emulation error is resolved by native execution and the single ARM64 file-stat alias correction.

The [problem dataset](https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro) is gated. Before `run`, accept its access/contact-sharing conditions using your Hugging Face account, then authenticate locally with `uv run hf auth login` (or configure `HF_TOKEN` in your environment). Do not put tokens in source code, result files, or chat. The testcase dataset is separate and was accessible for the judge smoke check. Without authorized problem-dataset access, the runner stops before model inference and reports how to authenticate.

`check` verifies Docker, model discovery, and a known correct Codeforces 2000A C++ solution before any inference. `run` repeats that check, records the resolved Hugging Face dataset revision, and evaluates all three endpoints sequentially with one attempt, temperature 0, and a 4,096-token budget. Request and judge timeouts default to 300 seconds. This budget is separate from the notebooks' short-answer settings; it can still truncate reasoning or solutions.

To exclude an unavailable vLLM server, use `python livecodebench.py run --models ollama spark`. The omitted model displays **Not run** in the presentation.

Options include `--judge-platform linux/arm64`, `--full`, `--limit 20`, `--models ollama spark`, `--max-tokens 8192`, `--dataset-revision <commit>`, `--request-timeout 600`, `--judge-timeout 600`, `--workers 2`, `--upstream <path>`, and `--output <directory>`. Use a fresh output directory for each run; existing results are never overwritten. To compare separately run models, use the same dataset revision, selection, settings, and judge architecture.

Results are checkpointed after each problem in `benchmark-results/{vllm,ollama,spark}.json`. They include full responses and extracted code, verdicts, inference latency, usage, finish reasons, model identity, upstream commit, dataset revision, and selected IDs. API keys are not saved. Interrupted runs retain completed rows; rerunning uses a new output directory rather than silently resuming or changing earlier results.

`render --results <directory> --output <file>` works without Docker or benchmark-only dependencies. By default it writes `presentation/livecodebench-results.tex`, which the deck includes before its summary. Missing models display **Not run**; partial runs retain all selected problems in the acceptance denominator. Comparison fails if problem IDs, revisions, or settings differ. Completed judging means a definitive code verdict; inference errors, missing code, and judge errors/timeouts are counted separately as failures. Truncated responses can also receive a code verdict. Mean inference latency includes valid generation responses, including responses without extractable code, and excludes judge time.

The generated section contains methodology, comparison, and verdict tables. Existing notebook metrics are unchanged. With no result files, the included slides explicitly show that evaluations have not run. Never substitute invented scores for unavailable endpoints or incompatible judging.

The 8 October 2026 workshop run used dataset revision `adebffce047dddb7768a86bace6aea4f7425e3bc`, native ARM64 judging, and the default 4,096-token budget. Spark accepted 3/10 solutions (four Wrong Answer, two Time Limit Exceeded, one Signalled); all 10 Ollama responses exhausted the token budget on reasoning without returning final code. Local vLLM was intentionally skipped and displays **Not run**. These outcomes describe this subset and configuration, rather than a general model ranking. The saved local JSON files are ignored by Git; the generated slides and PDF retain their aggregated results.

The presentation now uses Spark's 20 October 2026 rerun at `http://spark-0240:9000`: 4/10 accepted, four Compilation Error and two Wrong Answer, with 25.32 s mean inference latency and no truncations. All 10 solutions were evaluated. Problem IDs, dataset revision and generation settings match the earlier run. Ollama's 8 October results remain in the comparison, with each run date shown on the slide. The latest Spark JSON is retained in `benchmark-results/spark-2026-10-20-port9000/` and promoted to `benchmark-results/spark.json` for default rendering; the earlier Spark JSON is archived in `benchmark-results/spark-2026-10-08-port8000/`.

## Development

```sh
uv run --group dev python -m pytest tests/test_livecodebench.py
```

Tests mock inference and judging; the live Docker smoke check validates compilation, sandbox execution, dataset test cases, and accepted verdict handling together.
