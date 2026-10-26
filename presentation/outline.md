# Presentation outline

- Slide 1

Title slide: "DIY-AI"
Subtitle: "Running Large Language and Agentic Models in your own environments"
Date: 26 october 2026.
Presenter: Dr. Coert van Gemeren, HU University of Applied Sciences Utrecht.

- Slide 2

General introduction explaining how typically the setup of LLMs works: a webservice (Ollama, vLLM) or a program (mlx-vlm) that can run trained models: mlx-community/gemma-4-e4b-it-4bit (vLLM, no audio), deepseek-ai/DeepSeek-V4-Flash (vLLM), openai/whisper-large-v3 (vLLM), gemma4:26b (Ollama), google/gemma-4-E4B-it (mlx-vlm, with audio).
The point I want to get across, is that there is a forest of frameworks and models that one needs to navigate to run local AI. However, hardware is more than ever capable of running it yourself, as long as you make smart choices about how and what you want to achieve.

- Slide 3

Introduce the four types of hardware/framework combinations that we will discuss: Ollama on Apple Silicon, vLLM on Apple Silicon, mlx-vlm on Apple Silicon and vLLM of DGX Spark (via VPN connection).
The aim is to show the ease of using LLMs locally. Models can be started at an instant. Ollama is the entry level, then vLLM locally and finally vLLM over a VPN connection (accessing a remote cluster).

- Slide 4

Introduce the three metrics that we will test: five questions (observed deployment performance, not a controlled model-speed comparison), curated audio transcription of Jeroen Hendrikse and finally the LiveCodeBench-Pro test. In the saved self-hosted coding comparison, only DeepSeek on the remote cluster produced judgeable code. Keep the local vLLM result labelled "Not run".

- Slide 5

Explain "Model size is not the same as inference cost" with the progression dense small model → sparse MoE → very large sparse MoE: Gemma 4 E4B → Gemma 4 26B A4B → DeepSeek V4 Flash.

| Model | Total parameters | Active/effective per token |
| --- | --- | --- |
| Gemma 4 E4B (dense) | ~8B including embeddings | ~4.5B effective |
| Gemma 4 26B A4B (MoE) | ~25.2B | ~3.8–4B active |
| DeepSeek V4 Flash (MoE) | ~284B | ~13B active |

Distinguish E4B's embedding lookups from MoE routing. Gemma 26B can outperform E4B in tokens/s because only about 4B parameters are active per token, not all 26B, but the workshop measurements do not isolate architecture as the cause. DeepSeek offers 284B parameters of model capacity with about 13B active per token. Total weights still require storage. Cite the [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4) and [DeepSeek release](https://api-docs.deepseek.com/news/news260424/).

- Slide 6

Show the original five-question snapshot, retaining 18.64 / 34.37 / 28.94 completion tokens/s as observed inference performance of three local deployments. Preserve the request latencies, with completion rate as the final table row. Keep completeness observations and output-limit details in the presenter notes. The metric is total reported completion tokens divided by total request time, including prompt processing and request overhead. Emphasize real-world local deployment trade-offs rather than intrinsic model speed.

- Slide 7

Explain why the measurements are not apples-to-apples. E4B uses the 4-bit MLX model through vLLM Metal on the M1 Pro; Gemma 4 26B A4B uses Ollama on the M1 Pro; DeepSeek V4 Flash uses distributed vLLM across two DGX Spark nodes. Serving stacks, hardware, formats/quantization, tokenizers and reasoning behavior differ. Do not invent exact quantization metadata absent from the historical record. Keep output-limit details and the newer Ollama run out of the visible slide; those caveats remain in the presenter notes.

- Slide 8

Show test results of the transcription task, using the three different models: openai/whisper-large-v3 (local vLLM), google/gemma-4-E4B-it (mlx-vlm) and Qwen/Qwen3-Omni-30B-A3B-Instruct (on 1x DGX Spark). Results are found in the relevant notebooks: notebooks/local-llm-whisper.ipynb, notebooks/local-vlm-gemma.ipynb and notebooks/local-llm-qwen-omni.ipynb

- Slide 9

Discuss WER (see original presentation) and make a point of the fact that google/gemma-4-E4B-it (mlx-vlm) outperformes Qwen/Qwen3-Omni-30B-A3B-Instruct slightly in terms of WER.

- Slide 10

Show the evaluation of the saved five-sentence English summary against the full Dutch Gemma transcription. The local `mlx-community/gemma-4-e4b-it-4bit` model both generated and evaluated the summary. Its subjective ratings were factual faithfulness 5/5, main-point coverage 5/5, preserving uncertainty 4/5, and English clarity 5/5.

Separate these ratings from the subsequent source checks: the evaluator incorrectly counted four sentences instead of five; the summary omits that the incidental ischemic finding was new since the previous MRI; "residual tumor" is clearer English than "rest tumor"; and the four-point check is compressed imprecisely. The summary is broadly faithful, but the evaluator also needs checking. Do not present these scores as an independent benchmark or clinical validation. Evidence: `notebooks/local-llm-gemma-summary.ipynb` and `validation/gemma-summary-evaluation-2026-10-26/evaluation.md` / `evaluation.json`.

- Slide 11

Introduce the LiveCodeBench-Pro challenge. Local MLX trials did not yield usable solutions; the saved Ollama run hit its 4,096-token limit in all ten responses without final code, while local vLLM was not run in the judged comparison. These observations expose task difficulty and token-budget limitations, not a general model capability limit. Introduce DeepSeek V4 Flash as a sparse MoE with ~284B total and ~13B active parameters per token, running on a Spark cluster (2x DGX Spark, total 256GB unified memory).

- Slide 12

Show the results of LiveCodeBench-Pro and discuss them. Contrast it to the results of this benchmark achievable with cloud models like GPT-6 Astra or Claude Opus 5.5 (lookup of metrics are available for these models). We tested LiveCodeBench-Pro with GPT-6 Astra, results in presentation/livecodebench-gpt-6-astra-results.tex

- Slide 13

Introduce a local RAG implementation for advanced document search using a DGX Spark. Introduce the framework, that roughly looks like:

```text
Documents
    │
    ▼
Chunking
    │
    ▼
EmbeddingGemma
    │
    ▼
Vector DB (like Milvus)
        │ Top-K
        ▼
BGE reranker v2(.5) Gemma2 lightweight (optional)
        │ Top-N
        ▼
Gemma 4 31B
        │
        ▼
      Answer
```

- Slide 14

Title: "Embeddings: meaning as coordinates". Explain embeddings as vectors (lists of numbers), with similar meanings tending to occupy nearby positions in a high-dimensional space. Introduce E(King) + E(Woman) - E(Man) approximately E(Queen), where E maps a word to its embedding vector, following [3Blue1Brown's lesson](https://www.3blue1brown.com/lessons/gpt/).

Make this an open-ended audience quiz: "Holland + Fries - Kroket approximately ? Which country?" Show an editable TikZ diagram with parallel displacements Kroket to Fries and Holland to the unknown country, both labelled + Fries - Kroket. On the next click, reveal Belgium and "Add the fries association and subtract the kroket association." Keep the layout fixed across the two overlays.

Caption the image "Illustrative 2D embedding space; not a measured model result." Explain in the presenter notes that the snack analogy is invented, approximate, and not a measured EmbeddingGemma result. Connect the intuition back to RAG: EmbeddingGemma represents questions and document chunks in a shared space for similarity-based retrieval. Cite the [EmbeddingGemma model card](https://huggingface.co/google/embeddinggemma-300m).

The deck has 15 logical slides and 16 PDF pages. Both quiz overlays display slide number 14 / 15.

- Slide 15

Finally there's a handover to the next presenter, who will introduce an Agentic orchestrator for use in clinical imaging practice.

Place the download QR code from `images/github-link.svg` prominently beside the handover, with the invitation "Try it after the presentation". Use a large, crisp code with a clear white margin so the audience can scan it. Generate both the SVG and the matching vector PDF, `images/github-link.pdf`, with `uv run scripts/generate-workshop-qr.py`. Both encode `https://github.com/uashogeschoolutrecht/IMAGINE-workshop-2026` directly and include a four-module white margin. Keep this slide visible long enough to scan; hands-on exploration follows the presentation.
