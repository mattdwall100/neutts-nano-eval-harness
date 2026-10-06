# NeuTTS-Nano on-device benchmark and optimisation

A benchmark harness for [NeuTTS-Nano](https://github.com/neuphonic/neutts) on a laptop CPU, and the optimisation it drove. As shipped, the model runs 10x slower than real time on this machine. The final configuration runs at 0.87x in batch and 0.95x streaming without stalls, with no retraining and, for the last step, byte-identical audio.

![headline](results/headline.png)

| # | Step added | RTF |
|---|---|---|
| 1 | As shipped: torch bf16, torch codec, 13 s reference, watermark | 10.6 |
| 2 | Q4 GGUF backbone, ONNX int8 codec | 2.27 |
| 3 | 5 s reference clip | 1.12 |
| 4 | Watermark off | 1.11 |
| 5 | llama.cpp compiled for this CPU | 1.08 |
| 6 | Output projection sliced to the speech rows (49-line llama.cpp patch) | **0.87** |
| 7 | Decode threads pinned to P-cores | 0.87 |
| 8 | Streaming, shipped streaming settings | 2.56, stalls |
| 9 | Streaming, codec thread spinning off, 50-frame chunks | **0.95**, no stalls |

RTF is seconds of compute per second of audio; below 1.0 is real time. One voice, one interleaved session, 20 utterances per row (row 1: 2). On five voices never used in tuning: batch 0.73, streaming 0.96, zero stalls in 99.

The reasoning, the figures and the limits are in **[DESIGN.md](DESIGN.md)**.

## Hardware

Intel Core i7-1265U (2 performance + 8 efficiency cores, 15 W), 32 GB RAM, no GPU used, Windows 11 on mains in "best performance" power mode, Python 3.12 via `uv`. Thread settings and the compiled library are specific to this chip; the harness and the two main findings are not.

## Setup

```bash
uv sync
```

Models download on first run. The `neuphonic/*` Hugging Face repos are gated: accept their terms on huggingface.co and run `huggingface-cli login` first (about 7 GB in total).

Tested on Windows. The `llama-cpp-python` wheel pinned in `pyproject.toml` is Windows-only; elsewhere `uv sync` builds it from source and needs a C++ compiler. The patched library in `llama_patch/lib-slice` is Windows x86-64; sweeps that need it say so and exit if it is missing.

## Run it

One command, about 8 minutes, on an idle laptop:

```bash
PYTHONUTF8=1 uv run harness/sweep.py quick
uv run harness/plot.py results/quick_*
```

That runs Neuphonic's recommended on-device setup against the final configuration in batch and streaming, scores both, and prints a table. Every sweep in DESIGN.md is a named set in `harness/sweep.py`; the full headline chain is `uv run harness/sweep.py h1 --interleave` (25 min) and the held-out test is `test`.

Each run writes `results/<name>/`: one row per utterance with every stage time, RTF, memory, interference from other processes, and for streaming the time to first audio and worst stall; a summary with medians, word error rate (Whisper) and speaker similarity (ECAPA); and the audio.

A single configuration:

```bash
uv run harness/bench.py --name mine --backbone neuphonic/neutts-nano-q4-gguf --codec neuphonic/neucodec-onnx-decoder-int8 \
    --ref-audio samples/jo5.wav --ref-text samples/jo5.txt --no-watermark --codec-no-spin --stream --chunk-frames 50
uv run harness/score.py results/mine
```

## The final configuration

| Component | Setting |
|---|---|
| Backbone | `neutts-nano-q4-gguf` on the patched llama.cpp, `LLAMA_OUTPUT_ROWS=128261:65537` |
| Codec | `neucodec-onnx-decoder-int8` with thread spinning off (`--codec-no-spin`) |
| Reference | a short clip, about 5 s, ending on a natural pause; check per voice |
| Watermark | off (a product decision for Neuphonic; 0.03 RTF in batch, large per chunk in streaming) |
| Streaming | 50-frame chunks, lookback 50; low-latency alternative 25 / 10 |
| Threads | 4 decode threads pinned to P-cores, 12 prefill (optional, within noise) |

## The patched llama.cpp

The model's vocabulary is 194k rows but it only ever emits the 65,536 speech codes plus the stop token, and the final projection over those rows is half of every decode step. The patch (`llama_patch/llama_output_rows.patch`) scores only a row range; over 17,498 measured steps no dropped row ever entered the top-50 sampling set, and the output is byte-identical.

To use the shipped Windows build, set `LLAMA_CPP_LIB_PATH` to the absolute path of `llama_patch/lib-slice` and `LLAMA_OUTPUT_ROWS=128261:65537`; the sweep driver does this for you.

To rebuild, or build for Linux or macOS: clone llama.cpp at `c0159f9c` (the commit `llama-cpp-python` 0.3.19 vendors, so the bindings match), apply the patch, and build the `llama` target as shared libraries with any C++17 compiler and CMake. I used portable LLVM-MinGW and CMake zips, no installer:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
      "-DCMAKE_CXX_FLAGS=-include algorithm -include new"    # LLVM-MinGW only
cmake --build build --target llama -j
```

Put `ggml`, `ggml-base`, `ggml-cpu` and `libllama` from `build/bin` in one folder (plus, with MinGW, its `libc++`, `libunwind`, `libomp` and `libwinpthread-1` runtime DLLs) and point `LLAMA_CPP_LIB_PATH` at it. `uv run experiments/slice_test.py` checks the build against itself with the slice off.

## Repository

| Path | Purpose |
|---|---|
| `harness/` | `bench.py` (the harness), `score.py`, `sweep.py`, `plot.py`, `plot_headline.py` |
| `experiments/` | microbenchmarks and one-off studies: threads, build grid, the slice evidence and test, a token-level streaming loop, reference-clip cutting |
| `llama_patch/` | the patch and the built Windows library |
| `samples/`, `corpus.txt` | Neuphonic's reference voices with cut variants; 70 Harvard-sentence texts (lines 1-10 used for tuning) |
| `results/` | tables, summaries, per-utterance rows and figures for every sweep (audio not committed) |
| `DESIGN.md`, `docs/` | the decisions; the brief and the original plan |
