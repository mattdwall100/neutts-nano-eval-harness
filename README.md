# NeuTTS-Nano on-device benchmark and optimisation

A benchmarking harness for [NeuTTS-Nano](https://github.com/neuphonic/neutts) on a laptop CPU, and the optimisation it drove: **the shipped configuration runs at 10-17x slower than real time on this machine; the final one runs at 0.87x in batch and 0.95x streaming, with the same model weights and byte-identical audio for the last step.**

![headline](results/headline.png)

| # | Step added (each row includes all above it) | RTF | Peak RAM |
|---|---|---|---|
| 1 | As shipped: torch bf16 backbone, torch codec, 13 s reference, watermark | 10.6 | 3.3 GB |
| 2 | Q4 GGUF backbone, ONNX int8 codec | 2.27 | 1.3 GB |
| 3 | 5 s reference clip | 1.12 | |
| 4 | Watermark off | 1.11 | |
| 5 | llama.cpp compiled for this CPU | 1.08 | |
| 6 | Output projection sliced to the speech rows (49-line llama.cpp patch) | **0.87** | |
| 7 | Decode threads pinned to P-cores | 0.87 | |
| 8 | Streaming with the shipped streaming settings | 2.56, stalls | |
| 9 | Streaming, codec thread spinning off, 50-frame chunks | **0.95, no stalls, first audio 1.4 s** | 1.4 GB |

RTF = seconds of compute per second of audio; below 1.0 is real time. One interleaved session, voice "jo", 20 utterances per row. Quality (word error rate, speaker similarity) is flat from row 2 on. On five unseen voices and unseen sentences the final configuration measures 0.73 batch and 0.96 streaming, stall-free (see [DESIGN.md](DESIGN.md), "Held-out test set").

Everything about *why* is in **[DESIGN.md](DESIGN.md)**: metric definitions, measurement controls, every sweep with its figure, the things that did not work, and the limits.

## Hardware

| | |
|---|---|
| CPU | Intel Core i7-1265U: 2 performance + 8 efficiency cores, 12 threads, 15 W |
| RAM | 32 GB |
| GPU | Intel Iris Xe only; no CUDA. Everything runs on CPU |
| OS | Windows 11 Pro, "best performance" power mode, on mains |
| Python | 3.12 via `uv` |

Thread counts, core pinning and the compiled llama.cpp are specific to this chip. The harness, the slice and the streaming fixes are not.

## Setup

```bash
uv sync
```

That installs the reference `neutts` package (which bundles espeak-ng), the prebuilt `llama-cpp-python` CPU wheel, ONNX Runtime, faster-whisper and SpeechBrain. First run downloads the models from Hugging Face; the `neuphonic/*` repos are gated, so accept their terms on huggingface.co and log in (`huggingface-cli login`) first. Repos used: `neutts-nano`, `neutts-nano-q4-gguf`, `neutts-nano-q8-gguf`, `neucodec`, `distill-neucodec`, `neucodec-onnx-decoder`, `neucodec-onnx-decoder-int8` (about 7 GB).

Tested on Windows. The `llama-cpp-python` wheel URL in `pyproject.toml` is Windows-only; on Linux or macOS `uv sync` builds it from source (needs a C++ compiler). The patched library (`llama_patch/lib-slice`) is Windows x86-64; rows 5-9 of the headline need it, everything else runs without it.

## Run the benchmark

One command, about 5 minutes, idle laptop:

```bash
PYTHONUTF8=1 uv run harness/sweep.py quick 2>&1 | tee results/quick_sweep.log
```

It runs the shipped Q4 + ONNX configuration and the final configuration in batch and streaming on 6 sentences x 2 seeds, scores them, and prints a table. Then:

```bash
uv run harness/plot.py results/quick_*        # stacked per-stage RTF, config x metric heatmap, streaming TTFA/stall
```

The full headline chain (25 min) and every sweep in DESIGN.md are named sets in `sweep.py`:

```bash
uv run harness/sweep.py h1 --interleave       # the nine-row chain above
uv run harness/sweep.py test                  # five unseen voices x 20 unseen texts, batch + streaming
uv run harness/sweep.py --table results/h1_*  # re-print a table;  uv run harness/plot.py results/h1_* --prefix h1
```

Each run writes `results/<name>/`: `meta.json` (config, hardware, power mode, preflight load), `rows.jsonl` (one row per utterance: every stage time, RTF, tokens, memory, other-process CPU, streaming TTFA and stall), `summary.json` (medians plus WER and speaker similarity), `scores.jsonl`, and the WAVs.

### What is measured

| Metric | Meaning |
|---|---|
| RTF, and per stage: phonemize, prompt, prefill, generate, codec decode, watermark, glue | where the time goes; glue (unaccounted) stays near zero |
| Decode tok/s, prefill tok/s | comparable with Neuphonic's published numbers |
| Peak RSS | memory ceiling |
| TTFA, worst stall (streaming) | time to first audio; whether playback would ever run dry |
| WER (faster-whisper small.en, compound-tolerant) | words intact? |
| Speaker similarity (ECAPA cosine vs the reference voice) | voice intact? |
| Seam distance (log-mel dB vs the batch waveform of the same seed) | chunking artefacts |
| `n_failed` | utterances where the model produced no speech tokens |

Trust controls: a preflight that refuses to start on a busy machine, per-row attribution of other processes' CPU with a `suspect` flag, high process priority, warm-up, paired seeds across configurations, and interleaved rounds so drift hits every configuration equally. DESIGN.md explains why there is deliberately no gate-and-retry.

### Run a single configuration

```bash
uv run harness/bench.py --name mine --backbone neuphonic/neutts-nano-q4-gguf --codec neuphonic/neucodec-onnx-decoder-int8 \
    --ref-audio samples/jo5.wav --ref-text samples/jo5.txt --no-watermark --codec-no-spin \
    --stream --chunk-frames 50 --lookback 50 --limit 10 --repeats 2
uv run harness/score.py results/mine
```

## The final configuration

| Component | Setting |
|---|---|
| Backbone | `neuphonic/neutts-nano-q4-gguf` on llama.cpp with `llama_patch/llama_output_rows.patch`, `LLAMA_OUTPUT_ROWS=128261:65537` |
| Codec | `neuphonic/neucodec-onnx-decoder-int8`, `session.intra_op.allow_spinning=0` |
| Reference | a short clip that **ends on a sentence boundary** (about 5 s); shorter is faster, mid-sentence cuts hurt quality |
| Watermark | off (a product decision; < 0.03 RTF in batch, large per chunk in streaming) |
| Streaming | 50-frame chunks, lookback 50, lookforward 5; low-latency alternative 25 / 10 (first audio 0.9 s) |
| Threads | 4 decode threads pinned to the P-cores, 12 prefill threads (within noise; optional) |

### The patched llama.cpp

`llama_patch/llama_output_rows.patch` (49 lines against llama.cpp `c0159f9c`, the revision `llama-cpp-python` 0.3.19 vendors) restricts the final projection to a row range of the vocabulary. NeuTTS-Nano's vocabulary is 194k rows but it only ever emits the 65,536 speech codes plus the stop token; over 17,498 measured decode steps the other rows never came within rank 335 of the sampling cut-off. The slice is a third less work per decode step and the tokens are identical for a given seed, so the audio is byte-identical. The built library is in `llama_patch/lib-slice`; use it with `LLAMA_CPP_LIB_PATH=<absolute path>` and switch the slice on with `LLAMA_OUTPUT_ROWS=128261:65537`. To rebuild: clone llama.cpp at that commit, `git apply llama_output_rows.patch`, build the `llama` target as shared libraries (CMake, any C++17 compiler; with LLVM-MinGW add `-DCMAKE_CXX_FLAGS="-include algorithm -include new"`), and put the DLLs next to the runtime DLLs.

## Repository

| File | Purpose |
|---|---|
| `bench.py` | the harness: wraps the reference `NeuTTS` class with stage timers; batch and streaming |
| `score.py` | WER, speaker similarity, seam distance |
| `sweep.py` | named configuration sets, fresh process per config, interleaving, tables |
| `plot.py` | stage, heatmap and streaming figures |
| `tune.py`, `grid.py` | backbone-only microbenchmarks for thread, build and wait-policy experiments |
| `slice_stats.py`, `slice_test.py` | the evidence for, and the test of, the output-row slice |
| `stream.py` | token-level streaming loop (experiment; kept, not used in the final configuration) |
| `make_refs.py` | cut a reference voice at a word boundary with a matching transcript |
| `corpus.txt` | 70 Harvard sentences; lines 1-10 were used for tuning, the rest are held out |
| `samples/` | Neuphonic's reference voices, cut variants, cached codes |
| `results/` | tables, summaries, rows, figures for every sweep (audio not committed) |
| `DESIGN.md` | decisions, findings, figures, limits |
| `docs/PLAN.md` | the original plan, kept as written |
