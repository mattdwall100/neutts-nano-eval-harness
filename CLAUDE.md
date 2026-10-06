# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

`harness/` (bench, score, sweep, plot), `experiments/` (tune, grid, slice_stats, slice_test, stream, make_refs), `llama_patch/` (patch + built DLLs), `docs/` (brief, plan). Run every script from the repo root: paths like `results/`, `samples/`, `corpus.txt` are relative to it.

## What this is

Take-home for Neuphonic (see `docs/BRIEF_NEUPHONIC.md`): a benchmarking harness for NeuTTS-Nano on-device TTS, then optimise it as far as possible on this laptop. `docs/PLAN.md` holds the agreed scope, metric definitions and tiered optimisation plan; read it before proposing new work. `DESIGN.md` (to be written) records what was actually done and why.

Hardware is fixed: Intel i7-1265U (2 P + 8 E cores), 32 GB, no CUDA, Windows 11. Everything runs on CPU.

## Commands

```powershell
# the sweep (fresh process per config, scores at the end, writes results/b1_table.md). ~20 min idle.
$env:PYTHONUTF8=1; uv run harness/sweep.py b1 2>&1 | Tee-Object results/b1_sweep.log
uv run harness/sweep.py b1 --interleave          # final numbers: round-robin repeats across configs
uv run harness/sweep.py --table results/b1_*     # re-print the table
uv run harness/plot.py results/b1_*              # results/b1_stages.png (stacked stage RTF) + b1_heatmap.png (config x metric)
uv run harness/sweep.py b2                       # B2: tuned threads/affinity and shorter references on the Q4 + ONNX int8 base
uv run experiments/tune.py threads|probe|table       # backbone-only microbench (no codec/ASR), +-5% variance; use for lever experiments
uv run harness/score.py --rescore results/b1_*   # recompute WER from stored transcripts after a scorer change
# one configuration
uv run harness/bench.py --name q4_onnx8 --backbone neuphonic/neutts-nano-q4-gguf --codec neuphonic/neucodec-onnx-decoder-int8 --limit 6 --repeats 2
# flags: --dtype fp32  --threads N  --no-watermark  --max-other-cpu 20 (preflight)  --suspect-margin 10  --no-preflight  --warmup 1  --append --repeat-offset r
```

Measurement controls (preflight refusal, per-row load attribution with `suspect` flags, high priority, warm-up, interleaving) are described in DESIGN.md. There is deliberately no gate/retry: see DESIGN.md for why it was removed. The machine must be idle and on "best performance" power mode for numbers to count; bench exits 2 from preflight otherwise. Judge a run by `n_suspect` and `other_cpu_pct_median` in its summary.

```powershell
# quality floor: WER (faster-whisper small.en) + speaker similarity (SpeechBrain ECAPA) over run dirs
uv run harness/score.py results/baseline_torch results/q4_onnx8      # writes scores.jsonl, merges aggregates into summary.json
```

`score.py` passes numpy arrays to faster-whisper, never paths: its PyAV path is broken against the installed `av`. SpeechBrain is loaded with `LocalStrategy.COPY` because Windows without Developer Mode cannot symlink. The ECAPA model caches under `.cache/` (gitignored). `asr_floor_wer_on_reference` in summaries is Whisper's own WER on the reference clip; treat it as the noise floor, and compare configs relative to each other on the same seeds rather than reading absolute WER.

`PYTHONUTF8=1` is required on Windows: espeak emits IPA and the cp1252 console otherwise raises `UnicodeEncodeError`. `sweep.py` sets it (and warning suppression) for child processes; set it in the shell for ad-hoc Python.

Harness self-check (no models needed): the nested-interval logic in `StageTimer.exclusive()` is the only non-trivial code; each interval is charged to its smallest enclosing interval only. An assert on a synthetic generate > prompt > phonemize nesting runs at every `bench.py` start, as the WER-kernel asserts do in `score.py`.

## Corpus

`corpus.txt` is 70 texts from the Harvard sentences (public domain, phonetically balanced, no digits so WER needs no number normalisation). Lines 1-60 are single sentences, one from each of the first 60 lists; lines 61-70 are three-sentence passages built from sentences not used in the short set. Full runs take `--repeats 3` (210 utterances); use `--limit` for sweeps and reserve the full corpus for the final shortlist.

## Environment gotchas

- Python must be 3.12 (`neutts` pins `<3.14`; `requires-python` is pinned to 3.12 so uv resolves the prebuilt `llama-cpp-python` Windows wheel from the URL in `[tool.uv.sources]`). Do not let uv build llama-cpp-python from source; there is no cmake/MSVC here.
- `torchao<0.14` and `datasets>=3.0` are deliberate pins. `neucodec` imports `torchtune` for one class (`RotaryPositionalEmbeddings`); torchtune breaks on newer torchao and on the ancient `datasets` it otherwise resolves to.
- espeak-ng is bundled inside the `neutts` wheel (DLL + data). No system install. Do not try winget/msiexec; they hang on an elevation prompt in this shell.
- All `neuphonic/*` HF repos are gated and individually approved on the user's account; the stored token works. Repos in use: `neutts-nano` (torch), `neutts-nano-q4-gguf`, `neutts-nano-q8-gguf`, `neucodec`, `distill-neucodec`, `neucodec-onnx-decoder`, `neucodec-onnx-decoder-int8`. All are already in the HF cache.
- Torch backbone loads as bfloat16 in the reference code. This CPU has no bf16 instructions (AVX2 only), so the stock config runs at ~3 tok/s. Treat fp32 as the real torch baseline.

## How the harness works

`bench.py` does not reimplement inference. It instantiates the reference `neutts.NeuTTS` and monkeypatches instance methods with timing wrappers (`StageTimer.wrap`). Stage boundaries map to reference methods:

| stage | torch path | GGUF path |
|---|---|---|
| phonemize | `_to_phones` | `_to_phones` |
| prompt | `_apply_chat_template` (called from `infer`) | `_ggml_prompt` (called inside `_infer_ggml`) |
| prefill | first `backbone.forward` with seq_len > 1 | first `backbone.eval` with > 1 token |
| generate | `_infer_torch` (exclusive of prefill) | `_infer_ggml` (exclusive of prefill) |
| decode | `_decode` | `_decode` |
| watermark | `watermarker.apply_watermark` | same |

Nesting differs between paths (prompt is inside generate on GGUF), so `exclusive()` subtracts nested intervals. `glue_s` = e2e minus the sum of stages and should stay near zero; if it grows, a stage boundary is missing. Token count is taken by counting `<|speech_` in the string passed to `_decode`; `tok_per_s` is decode-only (generate excl. prefill). Prefill is ~3.4 s per utterance on Q4 and is re-done every call because the reference code calls `backbone.reset()`.

Seeding: `tts._seed = seed_base + repeat` before every `infer`, so every configuration uses the identical seed set and comparisons are paired. The `NeuTTS` constructor's `seed` arg is not used because it would fix one seed for all calls.

Reference voice codes are cached as `samples/<voice>.pt` next to the wav. ONNX decoders cannot encode, and the torch encoder takes ~75 s, so always use the cache. `samples/jo.pt` and `samples/dave.pt` came from Neuphonic's repo.

Peak RSS uses `peak_wset` on Windows, which is a process-lifetime high-water mark, so it reflects load + inference, not inference alone.

## References

`samples/jo.wav` (13.1 s, 653 codes), `jo8.wav` (8.5 s, 432), `jo5.wav` (5 s, 252): cuts of the same clip at Whisper word boundaries with matching transcripts, each with cached `.pt` codes. Speaker similarity for short-reference runs is scored against the full clip via `--sim-ref` (stored as `sim_ref_audio` in meta).

## Threads and cores

Logical CPUs 0-3 are the P-cores (probe: 47 tok/s single-thread vs 26 on E-cores 4-11). `--threads` = llama.cpp decode threads, `--threads-batch` = prefill threads, `--decode-cores 0,1,2,3` pins decode steps to P-cores while prefill uses all cores (affinity switched inside the eval wrapper). Best found: 4/10 with decode on P-cores.

## Streaming

Only GGUF backbones support `infer_stream`. `bench.py --stream --chunk-frames 25 --lookforward 5 --lookback 50 --overlap 1` (reference defaults) sets the instance attributes and records per row: `ttfa_s` (first chunk yielded), `n_chunks`, `stall_s` (worst lateness of any chunk vs playback starting at TTFA; 0 = never stalls). In stream mode `generate` is the residual of e2e minus the other stages (the generator cannot be wrapped), so it includes generator overhead and glue is 0 by construction; tokens = audio frames. Each chunk re-decodes lookback+chunk+lookforward frames and watermarks per chunk, so codec and watermark cost scale with chunk count. `score.py --seam-ref results/<batch_run>` adds seam distortion (aligned log-mel distance in dB, 0 = identical): same seeds give identical tokens, so streamed-vs-batch waveform difference is purely chunking. `uv run harness/sweep.py s1` is the shipped-streaming sweep; `s2` is the ladder on our own loop.

`stream.py` is the only inference code we wrote ourselves: a token-level loop over `backbone.generate()` (no detokenise/stop-string per token), growing chunk schedule, optional watermark, optional KV-prefix reuse. It must prepend BOS like `create_completion` does; with that it is token-identical to the reference for a seed (`uv run experiments/stream.py check`, 12/12). It also fixes a reference bug: the tail chunk is placed at its true offset, so output length equals batch (the reference is one frame longer). Harness flags: `--stream-impl ours --chunk-schedule 10,25,50,100 --lookback 25 --reuse-prefix`. The vocoder has no receptive-field plateau (`results/vocoder_receptive_field.json`: 4.2 dB at lookback 0 -> 2.5 dB at 100), so lookback is a cost/quality dial; 25 is the knee.

## Patched llama.cpp (output-row slice)

`tools/` (gitignored) holds the portable toolchain (LLVM-MinGW, CMake, Ninja), the llama.cpp checkout at c0159f9c with the patch applied, and the experimental builds; the shipped build is `llama_patch/lib-slice` (committed). Use it with `LLAMA_CPP_LIB_PATH=<absolute path to llama_patch/lib-slice>` (must be absolute) and switch slicing on with `LLAMA_OUTPUT_ROWS=128261:65537`. Build needs `-DCMAKE_CXX_FLAGS="-include algorithm -include new"` (libc++ strictness) and only the `llama` target (the httplib helper does not compile under MinGW and is not needed); copy the MinGW runtime DLLs next to the built ones. llama.cpp reuses its compute graph when the batch shape repeats, so a toggle only takes effect on the next shape change. `uv run experiments/slice_test.py` checks score equality, speed and token identity; `uv run harness/sweep.py f1 --interleave` is the end-to-end comparison (sets both variables per config).

## Results

`results/` is gitignored for now. Each run dir has `meta.json` (config, hardware, power mode, preflight), `rows.jsonl` (one row per utterance incl. other-CPU% and the `suspect` flag), `summary.json` (medians, RTF quartiles, merged with scores), `scores.jsonl`, and WAVs named `t<text>_r<repeat>.wav`. The first B1 attempt was discarded as contaminated (see DESIGN.md).
