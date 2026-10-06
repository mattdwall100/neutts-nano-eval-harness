# NeuTTS-Nano on-device benchmark and optimisation

A benchmarking harness for [NeuTTS-Nano](https://github.com/neuphonic/neutts) on a laptop CPU, and the optimisation it drove: **the shipped configuration runs at 10-17x slower than real time on this machine; the final one runs at 0.87x in batch and 0.95x streaming. No retraining anywhere: only Neuphonic's published artefacts (the Q4 backbone, the int8 codec), and for the last optimisation step the audio is byte-identical.**

![headline](results/headline.png)

| # | Step added (each row includes all above it) | RTF | Peak RAM |
|---|---|---|---|
| 1 | As shipped: torch bf16 backbone, torch codec, 13 s reference, watermark | 10.6 | 3.3 GB |
| 2 | Q4 GGUF backbone, ONNX int8 codec | 2.27 | 1.0 GB |
| 3 | 5 s reference clip | 1.12 | |
| 4 | Watermark off | 1.11 | |
| 5 | llama.cpp compiled for this CPU | 1.08 | |
| 6 | Output projection sliced to the speech rows (49-line llama.cpp patch) | **0.87** | |
| 7 | Decode threads pinned to P-cores | 0.87 | |
| 8 | Streaming with the shipped streaming settings (on row 7's pinned config; unpinned it measures 1.69) | 2.56, stalls | |
| 9 | Streaming, codec thread spinning off and 50-frame chunks (two changes) | **0.95, no stalls, first audio 1.4 s** | |

RTF = seconds of compute per second of audio; below 1.0 is real time. One interleaved session, voice "jo", 20 utterances per row (row 1: 2). Rows 2-7 are the identical weights, a 2.6x gain. Word error rate stays at 2-6 % throughout; speaker similarity is 0.47-0.50 at 13 s and 5 s references in this run, though the 5 s clip measured slightly lower (0.42 vs 0.49) in the dedicated reference sweep. Peak RAM is 1.0 GB for every Q4 + ONNX row (a 1.3 GB figure in `h1_table.md` is a load-time artefact of the `gguf` package, since removed). On five voices never used in tuning, the final configuration measures 0.73 batch and 0.96 streaming with no stalls, with the caveats in DESIGN.md's test-set section.

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

Tested on Windows. The `llama-cpp-python` wheel URL in `pyproject.toml` is Windows-only; on Linux or macOS `uv sync` builds it from source (needs a C++ compiler). The patched library (`llama_patch/lib-slice`) is Windows x86-64. The `quick`, `h1` (rows 5-9), `test`, `f1`, `s3`, `s4` and `g1` sweeps need it and exit with a message if it is missing; `b1`, `b2`, `r1`, `s1`, `s2` and any `bench.py` run without `LLAMA_CPP_LIB_PATH` use the prebuilt wheel.

## Run the benchmark

One command, about 8 minutes including scoring, idle laptop:

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
| Reference | a short clip, about 5 s, ending on a natural pause: the headline and quick runs use `jo5` (ends on a comma); on two test voices a mid-sentence 5 s cut was much worse than the whole clip, see DESIGN.md |
| Watermark | off (a product decision; < 0.03 RTF in batch, large per chunk in streaming) |
| Streaming | 50-frame chunks, lookback 50, lookforward 5; low-latency alternative 25 / 10 (first audio 0.9 s) |
| Threads | 4 decode threads pinned to the P-cores, 12 prefill threads (0-5 % end to end, within noise; optional, and it interacts badly with the codec's spinning threads unless those are off) |

### The patched llama.cpp

`llama_patch/llama_output_rows.patch` restricts llama.cpp's final projection to a row range of the vocabulary. NeuTTS-Nano's vocabulary is 194k rows but it only ever emits the 65,536 speech codes plus the stop token; over 17,498 measured decode steps no other row ever entered the top-50 sampling set (the best any reached was rank 335 overall). The slice is a third less work per decode step, the tokens are identical for a given seed, so the audio is byte-identical. It is off by default; the row range comes from an environment variable.

**Using the shipped build (Windows x86-64 only).** `llama_patch/lib-slice/` holds the built DLLs plus the MinGW runtime. The Python package picks them up through two environment variables, which `sweep.py` sets for you per configuration:

```bash
LLAMA_CPP_LIB_PATH=C:/absolute/path/to/llama_patch/lib-slice   # must be absolute
LLAMA_OUTPUT_ROWS=128261:65537                                 # <|SPEECH_GENERATION_END|> + the 65,536 speech codes
```

**Rebuilding it, or building for Linux / macOS.** The patch is 49 lines in two files (`src/models/llama.cpp`, `src/llama-context.cpp`) and applies to llama.cpp commit `c0159f9c1f874da15e94f371d136f5920b4b5335`. That commit matters: it is the one `llama-cpp-python` 0.3.19 vendors, so the Python bindings match the library's ABI. A different llama-cpp-python version needs its own vendored commit (`git -C vendor/llama.cpp rev-parse HEAD` in that repo) and the patch may need re-basing.

| Requirement | What I used | Notes |
|---|---|---|
| C++17 compiler | LLVM-MinGW 20260922 (clang 21), portable zip | no installer or admin rights needed; MSVC or GCC also work |
| CMake >= 3.14 | CMake 4.4.4, portable zip | |
| Build tool | Ninja 1.13 | optional; any CMake generator |
| OpenMP | bundled with LLVM-MinGW (`libomp`) | measured faster than llama.cpp's own thread pool on this CPU; keep `GGML_OPENMP=ON` |

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout c0159f9c1f874da15e94f371d136f5920b4b5335
git apply ../llama_patch/llama_output_rows.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++       -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF       "-DCMAKE_CXX_FLAGS=-include algorithm -include new"      # LLVM-MinGW/libc++ only: two headers this revision forgets to include
cmake --build build --target llama -j
```

Build only the `llama` target: the bundled HTTP helper does not compile under MinGW and nothing here uses it. Then collect, into one folder: `build/bin/` `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `libllama.dll` (Linux: the matching `.so` files; macOS: `.dylib`), and with LLVM-MinGW the runtime DLLs from `<llvm-mingw>/x86_64-w64-mingw32/bin/` (`libc++.dll`, `libunwind.dll`, `libomp.dll`, `libwinpthread-1.dll`). Point `LLAMA_CPP_LIB_PATH` at that folder.

`GGML_NATIVE=ON` compiles for the CPU doing the build (worth about 7 % here); use `OFF` for a portable library. Two things to know when testing: llama.cpp reuses its compute graph while the batch shape repeats, so toggling `LLAMA_OUTPUT_ROWS` inside one process only takes effect at the next prompt; and `uv run experiments/slice_test.py` checks score equality, token identity and speed against the same binary with the slice off.

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
| `corpus.txt` | 70 texts from the Harvard sentences (60 single sentences, 10 three-sentence passages); lines 1-10 were used for tuning |
| `samples/` | Neuphonic's reference voices, cut variants, cached codes |
| `results/` | tables, summaries, rows, figures for every sweep (audio not committed) |
| `DESIGN.md` | decisions, findings, figures, limits |
| `docs/PLAN.md` | the original plan, kept as written |
