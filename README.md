# DIY-AI: run the workshop on your own computer

This repository accompanies the [workshop slides](presentation/pdfs/local-ai-workshop.pdf) ([LaTeX source](presentation/local-ai-workshop.tex)). You will install a model runtime, run a local language model, and use notebooks to test text generation and Dutch audio transcription.

The two supported local platforms are **Apple Silicon Macs (M1/M2/M3/M4/Mx)** and **Linux with an NVIDIA CUDA GPU**. The Spark exercises use servers supplied by the presenter. The RAG architecture and agentic handover are discussion topics in the slides; this repository does not deploy those systems.

## 1. Choose your route

You do not need to install every runtime. Start with Ollama for text chat; add vLLM and the audio exercises when ready.

| Your computer | Runtime | Workshop model | Exercise |
| --- | --- | --- | --- |
| Apple Silicon or Linux CUDA | Ollama | `gemma4:26b` | Five text questions |
| Apple Silicon | vLLM Metal | `mlx-community/gemma-4-e4b-it-4bit` | Five text questions through an HTTP API |
| Linux CUDA | vLLM | `google/gemma-4-E4B-it` | Five text questions and Gemma audio transcription |
| Apple Silicon | MLX-VLM | `google/gemma-4-E4B-it` | Gemma audio transcription directly in Python |
| Apple Silicon or Linux CUDA | vLLM Metal or CUDA vLLM | `openai/whisper-large-v3` | Whisper audio transcription |
| Either computer, with workshop VPN | Remote vLLM on DGX Spark | DeepSeek V4 Flash / Qwen3-Omni | Optional text, audio, and coding exercises |

A **model** is downloaded weights. A **runtime** loads those weights and performs inference. Ollama and vLLM run as **servers**: the notebook sends them HTTP requests. MLX-VLM runs directly inside the Gemma notebook on your Mac. On Linux, vLLM handles Gemma E4B's image and audio processing, so you do not need MLX-VLM there. This workshop uses vLLM Metal for Mac text/Whisper and MLX-VLM for Mac Gemma audio; this is a chosen configuration, not a claim that every vLLM Metal release lacks multimodal support.

### Memory matters more than the M-series generation

A model that fits a large-memory Mac may not fit an 8 GB or 16 GB Mac. `gemma4:26b` downloads roughly 16–19 GB depending on the variant; inference also needs working memory, context storage, and space for the operating system. Full-precision Gemma E4B also needs substantially more memory than the 4-bit MLX text model. NVIDIA VRAM and Mac unified memory are different resources; check what your machine actually has.

For a smaller **Ollama text** exercise, use `gemma4:e2b` and change the notebook model to match. Start one model at a time. A smaller model, different quantization, or different token budget produces a different experiment from the slides. See the [Ollama model sizes](https://ollama.com/library/gemma4) and [MLX-VLM model guidance](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md).

## 2. Prepare your computer

You need an internet connection for software and the first model downloads. Allow several GB for Python environments and tens of GB for model caches. Downloads are cached; the first launch takes longer than later launches.

### Apple Silicon

Use a native Terminal, not a terminal running under Rosetta:

```sh
uname -m
sw_vers -productVersion
```

The architecture should be `arm64`. Use **macOS 15 or newer** for the vLLM Metal route; check [its current requirements](https://github.com/vllm-project/vllm-metal) before installing. Ollama has its own [macOS requirements](https://docs.ollama.com/macos).

Install Apple's Command Line Tools for Git:

```sh
xcode-select --install
```

Complete the installer if prompted. macOS already supplies curl. Verify:

```sh
git --version
curl --version
```

For audio exercises, install [Homebrew](https://brew.sh/) using its installation instructions, including the printed PATH setup, then:

```sh
brew install ffmpeg
ffmpeg -version
```

### Linux with NVIDIA CUDA

Use your distribution's package manager to install **Git**, **curl**, and **CA certificates**. Install **ffmpeg** for the Gemma and Qwen audio exercises. Package names and commands vary by distribution; [FFmpeg's download page](https://ffmpeg.org/download.html) links to Linux packages.

Install the NVIDIA driver using your distribution's instructions or [NVIDIA's driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/). Reboot if the driver installer requires it, then verify:

```sh
git --version
curl --version
nvidia-smi
```

`nvidia-smi` must display your GPU without an error. vLLM requires a supported GPU (currently compute capability 7.5 or newer) and a driver compatible with its PyTorch/CUDA wheel. Check the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for the current requirements.

For the prebuilt-wheel route used here, you do **not** need to install the full CUDA Toolkit or have `nvcc` on PATH. The installer selects the PyTorch backend and verifies a real CUDA computation. The “CUDA Version” shown by `nvidia-smi` describes driver compatibility, not an installed compiler.

### Clone the repository and install Python

Run these commands in a directory where you keep projects:

```sh
git clone https://github.com/uashogeschoolutrecht/IMAGINE-workshop-2026.git
cd IMAGINE-workshop-2026
```

These instructions describe the cleaned workshop branch. Until it is merged, select it after cloning:

```sh
git switch final
```

Install [uv](https://docs.astral.sh/uv/) and the workshop's Python 3.12:

```sh
./scripts/install-python.sh
```

The script downloads a prebuilt Python through uv. It does not compile Python, change the system Python, or configure pyenv. If uv was just installed, open a new terminal and return to the repository, or load its PATH setup in the current shell:

```sh
. "$HOME/.local/bin/env"
uv --version
uv python find 3.12
```

All remaining commands assume the repository root is your working directory.

## 3. Set up and launch the notebooks

Install the shared notebook environment:

```sh
uv sync --locked
uv run --no-sync python -m ipykernel install --sys-prefix \
  --name workshop-local-llm --display-name "DIY-AI workshop (Python 3.12)"
```

`uv sync` creates `.venv` with JupyterLab, the notebook kernel, requests, pandas, and the other notebook dependencies. You do not need to activate it.

| Environment | Purpose | How it is installed |
| --- | --- | --- |
| `.venv` | JupyterLab and every notebook; optional direct MLX-VLM inference | `uv sync --locked` |
| `.venv-vllm` | Linux CUDA vLLM server only | CUDA installer below |
| `~/.venv-vllm-metal` | Apple Silicon vLLM Metal server only | Metal installer below |
| Ollama application/service | Ollama inference and model storage | Ollama installation below |

Keep server dependencies separate from notebook dependencies, even though both use Python 3.12. **Always select the notebook `.venv` kernel**, including when a vLLM server is running in another environment.

Use **terminal A** for the model server and **terminal B** for Jupyter. In terminal B:

```sh
uv run --no-sync jupyter lab
```

Open the localhost URL printed by Jupyter, open a notebook from `notebooks/`, select **DIY-AI workshop (Python 3.12)**, then use **Kernel → Restart Kernel and Run All Cells**. Read the first setup cell before running it: this is where the model, endpoint, and generation settings are configured.

`--no-sync` uses the environment you prepared. This matters on Mac: a later plain `uv sync` can remove the optional MLX-VLM group. After changing groups, restart the notebook kernel. When reopening the project, repeat the appropriate `uv sync` command and the Jupyter command.

Alternatively, install [VS Code](https://code.visualstudio.com/) and its Python and Jupyter extensions (recommended by this repository). Open the repository folder, open a notebook, and use **Select Kernel → Python Environments → `.venv/bin/python`**. Do not select either vLLM server environment.

Notebook environment variables must be set **before launching Jupyter or VS Code**. Exporting a variable in another terminal does not change a running kernel. You can instead edit the setup cell and rerun it. Base URLs can include a trailing `/v1`; notebooks normalize it. API keys are optional for local servers and belong in environment variables, not saved notebook cells.

## 4. Apple Silicon: install and run models

### Ollama: start here

1. Download [Ollama for macOS](https://ollama.com/download/mac), move it to Applications, and open it.
2. Follow its prompt to make the `ollama` command available. Open a new terminal if needed.
3. In terminal A, from the repository root:

```sh
ollama --version
./scripts/start-ollama.sh
```

The launcher reuses the app's server or starts one, pulls `gemma4:26b`, loads it, and checks GPU use. If it started the server itself, keep that terminal open. If the app already owns the server, the launcher returns when ready.

For a smaller model, use this **instead**:

```sh
./scripts/start-ollama.sh gemma4:e2b
```

Check the server and try interactive chat:

```sh
curl --fail http://localhost:11434/v1/models
ollama ps
ollama run gemma4:26b
```

Use `gemma4:e2b` in the last command if that is what you loaded. Type `/bye` to leave interactive chat. In terminal B, launch Jupyter and open [the Ollama notebook](notebooks/local-llm-ollama.ipynb). For the smaller model, launch with:

```sh
OLLAMA_MODEL=gemma4:e2b uv run --no-sync jupyter lab
```

When finished, unload whichever model you used:

```sh
ollama stop gemma4:26b
# Or: ollama stop gemma4:e2b
```

Use Ctrl+C in terminal A if the launcher owns the server, or quit the Ollama app if you want its service stopped. Unloading the model releases memory for the next exercise.

### vLLM Metal: a local HTTP model server

Install the upstream **stable** release into its separate environment:

```sh
./scripts/install-mlx-vllm.sh
```

The same installer updates an existing installation. In terminal A:

```sh
./scripts/start-mlx-vllm.sh --host 127.0.0.1
```

This serves `mlx-community/gemma-4-e4b-it-4bit` on port 8000. Wait for the server-ready message. In terminal B:

```sh
curl --fail http://localhost:8000/v1/models
uv run --no-sync jupyter lab
```

Open [the local vLLM notebook](notebooks/local-llm-vllm.ipynb). On Apple Silicon its default model matches this launcher. Stop the server with **Ctrl+C** in terminal A before changing models.

To run Whisper, stop Gemma and start:

```sh
./scripts/start-mlx-vllm.sh openai/whisper-large-v3 --host 127.0.0.1
```

Check `/v1/models` again, prepare the [audio files](#6-audio-files), and run [the Whisper notebook](notebooks/local-llm-whisper.ipynb). Whisper uses `/v1/audio/transcriptions`; it is a separate model, not an extra feature enabled on a running Gemma server. If your Metal release cannot serve Whisper, check [the project's current speech-to-text support](https://github.com/vllm-project/vllm-metal); the same notebook can call a CUDA Whisper server by changing `VLLM_BASE_URL`.

Both vLLM launchers accept `MODEL` followed by normal `vllm serve` arguments. For example, to use a different port:

```sh
./scripts/start-mlx-vllm.sh mlx-community/gemma-4-e4b-it-4bit --port 8001
```

Set `VLLM_BASE_URL=http://localhost:8001` in the notebook or before launching Jupyter.

### MLX-VLM: Gemma audio inside the notebook

Stop other model servers and unload Ollama models first. Install the optional Mac dependencies:

```sh
uv sync --locked --group mlx-vlm
uv run --no-sync jupyter lab
```

Prepare the [audio files](#6-audio-files), then open [the Gemma notebook](notebooks/local-vlm-gemma.ipynb). Keep `BACKEND` set to `mlx` and run all cells. Its model-load cell downloads and loads `google/gemma-4-E4B-it`; no HTTP server is needed. The next cell transcribes chunks, and the last cells calculate WER.

Shut down the notebook kernel to release the model memory. Closing its browser tab alone does not stop the kernel.

There is also a one-shot command for a **short clip of at most 28 seconds**:

```sh
./scripts/start-mlx-vlm.sh /absolute/path/to/short-clip.wav
```

Pass a prompt as the second argument if desired. Use the notebook for the full workshop recording because it splits longer audio automatically.

## 5. Linux CUDA: install and run models

### Ollama

Install the Linux service:

```sh
./scripts/install-ollama.sh
```

The official installer may request sudo privileges. Then in terminal A:

```sh
./scripts/start-ollama.sh
```

The launcher verifies the NVIDIA driver, starts or reuses Ollama, downloads and loads `gemma4:26b`, and verifies GPU acceleration. The smaller-model alternative is `./scripts/start-ollama.sh gemma4:e2b`.

In terminal B:

```sh
curl --fail http://localhost:11434/v1/models
ollama ps
uv run --no-sync jupyter lab
```

Open [the Ollama notebook](notebooks/local-llm-ollama.ipynb) and make its model match the launcher. Use `ollama run MODEL` for interactive chat and `ollama stop MODEL` to unload it. If the launcher started the server, Ctrl+C stops it. If the installer created a systemd service, the service remains running; `sudo systemctl stop ollama` stops that service when needed.

### vLLM: Gemma text and audio

Install the CUDA server environment:

```sh
./scripts/install-cuda-vllm.sh
```

The script creates `.venv-vllm`, installs vLLM with its audio extras and compatible PyTorch backend, checks the GPU's compute capability, and performs a CUDA tensor operation. It prints the installed version and detected GPU. It does not install GPU drivers.

In terminal A, start Gemma with a modest context budget for the workshop:

```sh
./scripts/start-cuda-vllm.sh --host 127.0.0.1 --max-model-len 4096 --max-num-seqs 1
```

This serves `google/gemma-4-E4B-it` on port 8000. First startup includes downloading weights and preparing the runtime. Wait until it reports that the API server is ready. In terminal B:

```sh
curl --fail http://localhost:8000/v1/models
uv run --no-sync jupyter lab
```

Run [the local vLLM notebook](notebooks/local-llm-vllm.ipynb) for text. For audio, prepare the files and run [the Gemma notebook](notebooks/local-vlm-gemma.ipynb) with `BACKEND="vllm"` (the Linux default). **Keep the same Gemma server running**: it handles the audio input as well as text. No MLX group or MLX package is needed on Linux.

For a different port or memory allocation, pass normal vLLM arguments, for example:

```sh
./scripts/start-cuda-vllm.sh google/gemma-4-E4B-it \
  --host 127.0.0.1 --port 8001 --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.8
```

Update the notebook's `VLLM_BASE_URL` to match. A smaller context reduces cache requirements; lowering memory utilization reserves less GPU memory for vLLM but can also leave too little for the model. Neither setting makes arbitrarily large weights fit.

### vLLM: Whisper

Stop the Gemma server with **Ctrl+C**, then start Whisper in terminal A:

```sh
./scripts/start-cuda-vllm.sh openai/whisper-large-v3 --host 127.0.0.1
```

Verify `/v1/models`, then run [the Whisper notebook](notebooks/local-llm-whisper.ipynb). Stop the server with Ctrl+C when finished.

## 6. Audio files

Gemma and Qwen use **ffmpeg** to decode the MP3 and split it into chunks. Install it using Homebrew on Mac or your Linux package manager, then verify:

```sh
ffmpeg -version
```

**Workshop download: [Audio recording and curated TXT/SRT references (ZIP)](https://filesender.surf.nl/?s=download&token=a2eafe0d-d472-4bc1-a497-1c6b70943524)**

The download page is **password protected**. Obtain the password from the presenter, open the link in your browser, enter the password when prompted, and download `UMC Utrecht - Jeroen Hendrikse - radiologieverslag.zip`. The recording and two curated reference transcripts are not included in Git.

You can keep the password locally in `.secrets/audio-download-password.txt`. The entire `.secrets/` directory is ignored by Git, so this file is not included when cloning the repository. To create it in a fresh checkout, run from the repository root:

```sh
mkdir -p .secrets
chmod 700 .secrets
touch .secrets/audio-download-password.txt
chmod 600 .secrets/audio-download-password.txt
```

Open the file in your editor and put only the download password on its first line, replacing any placeholder. Copy it into the download page when prompted; the notebooks do not read this file. Keep the password out of tracked files and share it separately from the repository.

From the **repository root** (the folder containing this README), run the following on Mac or Linux after setting up the notebook environment in section 3. This assumes the ZIP was saved in `~/Downloads`; adjust the ZIP path if your browser used a different name or location. Keep the quotes because the paths contain spaces.

```sh
mkdir -p "data/UMC Utrecht - Jeroen Hendrikse - radiologieverslag"
uv run --no-sync python -m zipfile -e \
  "$HOME/Downloads/UMC Utrecht - Jeroen Hendrikse - radiologieverslag.zip" \
  "data/UMC Utrecht - Jeroen Hendrikse - radiologieverslag"
```

The ZIP contains the files directly, so extract into that named folder, not just `data/` or `notebooks/`. If you prefer Finder or a file manager, extract the ZIP and place the three files in the same destination. The required layout is:

```text
data/
└── UMC Utrecht - Jeroen Hendrikse - radiologieverslag/
    ├── UMC Utrecht - Jeroen Hendrikse - radiologieverslag.mp3
    ├── UMC Utrecht - Jeroen Hendrikse - radiologieverslag - curated.txt
    └── UMC Utrecht - Jeroen Hendrikse - radiologieverslag - curated.srt
```

Verify the extracted files:

```sh
ls "data/UMC Utrecht - Jeroen Hendrikse - radiologieverslag"
```

You should see the MP3, curated TXT, and curated SRT shown above. An extra `__MACOSX` folder contains archive metadata and can be ignored. Avoid an extra nested folder with the same recording name: the three files must sit directly in the folder shown. The notebooks check for all three files before inference. Text exercises work without these files.

Gemma and Qwen decode to mono 16 kHz WAV and split at pauses where possible, with a maximum of 28 seconds per chunk. Whisper receives the full MP3. Each notebook prints the transcription and compares it with the curated TXT and SRT using **word error rate**:

`WER = (substitutions + deletions + insertions) / reference words`

References are used only for scoring. To use your own recording, edit the three paths and supply matching transcripts; the workshop's reference text cannot score a different recording meaningfully.

## 7. Which notebook should I run?

Every notebook uses **DIY-AI workshop (Python 3.12)** from `.venv`. Run its setup, model discovery/load, inference, and evaluation cells from top to bottom.

| Notebook | Required running model / launcher | Default endpoint or backend | Extra preparation |
| --- | --- | --- | --- |
| [Ollama text](notebooks/local-llm-ollama.ipynb) | `start-ollama.sh` → `gemma4:26b` | `localhost:11434` | Match `OLLAMA_MODEL` if you loaded a smaller model |
| [vLLM text](notebooks/local-llm-vllm.ipynb) | `start-mlx-vllm.sh` or `start-cuda-vllm.sh` → platform's Gemma model | `localhost:8000` | Keep the server terminal open |
| [Gemma transcription](notebooks/local-vlm-gemma.ipynb) | Mac: loaded by notebook; Linux: `start-cuda-vllm.sh` → `google/gemma-4-E4B-it` | Mac: `mlx`; Linux: `vllm` at `localhost:8000` | Audio + references, ffmpeg; `mlx-vlm` group only on Mac |
| [Whisper transcription](notebooks/local-llm-whisper.ipynb) | Either vLLM launcher with `openai/whisper-large-v3` | `localhost:8000` | Audio + references; stop Gemma on this port first |
| [Spark text](notebooks/local-llm-spark-cluster.ipynb) | Presenter-hosted `deepseek-ai/DeepSeek-V4-Flash` | `spark-0240:9000` | Presenter-provided VPN, server address and optional API key |
| [Qwen transcription](notebooks/local-llm-qwen-omni.ipynb) | Presenter-hosted `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `spark-0240:8000` | VPN, audio + references, ffmpeg |

Spark means **NVIDIA DGX Spark**, not Apache Spark or PySpark. You do not need a local model runtime to call a remote server; the normal notebook environment is sufficient. The private hostnames work only on the workshop network. Confirm availability and addresses with the presenter; the README does not provision the remote cluster.

| Notebook | Environment overrides (or edit its setup cell) |
| --- | --- |
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY` |
| vLLM text | `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY` |
| Gemma | `GEMMA_BACKEND` (`mlx` or `vllm`), `GEMMA_MODEL`; for vLLM: `VLLM_BASE_URL`, `VLLM_API_KEY` |
| Whisper | `VLLM_BASE_URL`, `WHISPER_MODEL`, `VLLM_API_KEY` |
| Spark text | `SPARK_BASE_URL`, `SPARK_MODEL`, `SPARK_API_KEY` |
| Qwen audio | `QWEN_BASE_URL`, `QWEN_MODEL`, `QWEN_API_KEY` |

The notebooks retain labeled historical outputs where relevant. **Saved output is not evidence that your current server has run successfully.** Restart and run all cells to generate a new result. The slides preserve their dated measurements; changing models or running notebooks does not update slide numbers automatically.

On **23 October 2026**, validation on an M1 Pro with 32 GB memory reproduced **12.00% Whisper WER** and **7.78% Gemma MLX-VLM WER** on the complete workshop recording against both matching curated references. Both text notebooks returned five visible answers; default token limits truncated three vLLM answers and two Ollama answers. The run used the installed vLLM Metal development build; the stable installer was not rerun. This validates that machine, without establishing a fresh-install or minimum-memory guarantee. Executed notebooks, logs and runtime details are retained locally in the ignored `validation/` directory.

## 8. Optional exercises and troubleshooting

### LiveCodeBench-Pro coding evaluation

This reproduces the type of C++ evaluation discussed in the presentation. It is optional and requires [Docker Desktop on Mac](https://docs.docker.com/desktop/setup/install/mac-install/) or [Docker Engine on Linux](https://docs.docker.com/engine/install/), running and accessible to your user. The judge uses a privileged container for nested resource controls, with seccomp enabled and architecture-specific configuration. No Docker installation is required for the text and audio notebooks.

Install the benchmark dependencies and the pinned upstream checkout:

```sh
uv sync --locked --group benchmark
mkdir -p .benchmark
git clone --recurse-submodules https://github.com/GavinZhengOI/LiveCodeBench-Pro.git .benchmark/LiveCodeBench-Pro
git -C .benchmark/LiveCodeBench-Pro checkout d614eef4f95b3654bf0758c64e1194b2575850b8
git -C .benchmark/LiveCodeBench-Pro submodule update --init --recursive
docker info
```

Accept access conditions for the [problem dataset](https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro) with your Hugging Face account, then authenticate locally:

```sh
uv run --no-sync hf auth login
```

Start the inference endpoint you want to test. The runner supports the same `VLLM_*`, `OLLAMA_*`, and `SPARK_*` endpoint/model/key overrides as the text notebooks. Its default vLLM model is the Mac MLX model; set `VLLM_MODEL=google/gemma-4-E4B-it` for CUDA. Explicitly choose available backends so you do not require all three servers:

```sh
uv run --no-sync python livecodebench.py check --models ollama
uv run --no-sync python livecodebench.py run --models ollama \
  --dataset-revision adebffce047dddb7768a86bace6aea4f7425e3bc \
  --output benchmark-results/my-ollama-run
```

For the workshop Spark server, use `--models spark`; the default is `http://spark-0240:9000`. `check` verifies model discovery and a known-correct C++ solution in the judge. `run` repeats the check and evaluates the first ten sorted unique problems with one attempt, temperature zero, and a 4,096-token budget. Initial Docker image and dataset downloads can take several minutes.

The judge defaults to native ARM64 on Apple Silicon and AMD64 on x86 Linux. Results are saved after each problem; choose a **new output directory** for every run. Do not overwrite the presentation's historical evidence. Compare runs only when problem selection, dataset revision, generation settings, and judge architecture match. See `uv run --no-sync python livecodebench.py run --help` for other options.

To export a run as a Beamer fragment:

```sh
uv run --no-sync python livecodebench.py render \
  --results benchmark-results/my-ollama-run \
  --output benchmark-results/my-ollama-run/results.tex
```

This fragment is not a standalone presentation and does not replace the current slides. The retained fragments in `presentation/` document the historical results. Raw historical JSON is not distributed in this checkout. The old full presentation remains on the `main` branch.

### Build the current presentation

Reading the supplied PDF requires no LaTeX installation. To rebuild, install a TeX distribution with **latexmk, Beamer, Latin Modern, booktabs, and PGF/TikZ**: [MacTeX](https://www.tug.org/mactex/) on Mac or [TeX Live](https://www.tug.org/texlive/) through your Linux distribution. Then:

```sh
./scripts/build-tex.sh presentation/local-ai-workshop.tex
```

The script keeps intermediates in a temporary directory and writes only `presentation/pdfs/local-ai-workshop.pdf`. The VS Code LaTeX Workshop recommendation uses this same script. The result fragments are inputs/evidence, not standalone TeX documents to build.

The closing slide's QR code links directly to `https://github.com/uashogeschoolutrecht/IMAGINE-workshop-2026`. To regenerate both `presentation/images/github-link.svg` and the matching vector PDF used by LaTeX, run:

```sh
uv run scripts/generate-workshop-qr.py
./scripts/build-tex.sh presentation/local-ai-workshop.tex
```

The Python script declares its own dependencies, which `uv` installs in an isolated environment. Both assets use the same QR matrix with a four-module white margin.

### Common problems

| Symptom | What to check |
| --- | --- |
| `uv: command not found` | Open a new terminal or source `~/.local/bin/env` after installation. |
| `ModuleNotFoundError` | Select `.venv/bin/python`; run `uv sync --locked` (add `--group mlx-vlm` for Mac Gemma), then restart the kernel. |
| An old Python 3.14 environment | `uv sync --locked` recreates `.venv` for Python 3.12; rerun kernel registration and select the new kernel. |
| Connection refused | Keep the model server running; wait for readiness; check its port and `/v1/models`. Jupyter itself is not the model server. |
| Port 8000 already in use | Stop your previous vLLM server, or launch the new one with `--port 8001` and update the notebook URL. |
| Model not found | Copy its exact ID from `/v1/models`; Ollama tags, MLX conversions, and Google model IDs are different. |
| Download returns 401/403 | Open the model/dataset page, complete any required access steps, and use `uv run --no-sync hf auth login`. Model-download credentials and server API keys serve different purposes. |
| CUDA unavailable | Check `nvidia-smi`, driver compatibility, and the installer's CUDA-operation check. Do not install packages into the system Python to fix the separate server environment. |
| Out of memory | Stop other servers; `ollama stop MODEL`; shut down MLX notebook kernels; try the smaller Ollama text model or reduce vLLM context. Full-precision weights still need to fit. |
| Missing recording or references | Follow the exact directory layout in section 6; the files are not included in Git. |
| `ffmpeg` missing | Install it with Homebrew or your Linux package manager, then restart Jupyter if PATH changed. |
| Answer cut short / transcription incomplete | Raise `MAX_TOKENS` within the server context limit and rerun; do not score a truncated transcript as complete. |
| Spark hostname unavailable | Connect to the workshop VPN and confirm the address with the presenter. Local exercises do not require Spark. |

### Development checks

```sh
uv run --locked --group dev python -m pytest -q
uv run --locked --group dev ruff check .
```

Tests mock inference and judging; they do not download models or prove hardware support.

Validation on 23 October 2026: all 72 tests and Ruff checks passed. The CUDA installer and Gemma E4B model discovery, text, image, and notebook audio requests were exercised on an NVIDIA RTX A5000 (24 GB) with vLLM 0.32.0. Audio used a short synthetic Dutch clip; the full workshop recording was unavailable. Apple Silicon dependency resolution and mocked backend tests passed, but Mac inference was not run on this Linux host. The presentation rebuilt successfully to eleven slides with its historical measurements retained.
