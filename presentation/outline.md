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

Introduce the three metrics that we will test: five questions (discussed on the previous version of the presentation), curated audio transcription of Jeroen Hendrikse and finally the LiveCodeBench-Pro test. Make clear the the LiveCodeBench-Pro only produces results using the largest model running on the remote cluster.

- Slide 5

Show results found in the original presentation of the first test: the five questions

- Slide 6

Show test results of the transcription task, using the three different models: openai/whisper-large-v3 (local vLLM), google/gemma-4-E4B-it (mlx-vlm) and Qwen/Qwen3-Omni-30B-A3B-Instruct (on 1x DGX Spark). Results are found in the relevant notebooks: notebooks/local-llm-whisper.ipynb, notebooks/local-vlm-gemma.ipynb and notebooks/local-llm-qwen-omni.ipynb

- Slide 7

Discuss WER (see original presentation) and make a point of the fact that google/gemma-4-E4B-it (mlx-vlm) outperformes Qwen/Qwen3-Omni-30B-A3B-Instruct slightly in terms of WER.

- Slide 8

Introduce LiveCodeBench-Pro challenge and make clear that the local mlx models that were tested were not powerful enough to produce any results on this test. Make clear that this shows that the programming challenges that are given are quite hard. Then introduce deepseek-ai/DeepSeek-V4-Flash that runs on a Spark cluster (2x DGX Spark, total 256GB unified memory).

- Slide 9

Show the results of LiveCodeBench-Pro and discuss them. Contrast it to the results of this benchmark achievable with cloud models like GPT-6 Astra or Claude Opus 5.5 (lookup of metrics are available for these models). We tested LiveCodeBench-Pro with GPT-6 Astra, results in presentation/livecodebench-gpt-6-astra-results.tex

- Slide 10

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

- Slide 11

Finally there's a handover to the next presenter, who will introduce an Agentic orchestrator for use in clinical imaging practice.
