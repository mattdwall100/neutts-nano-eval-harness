# Design decisions

Headlines only. Every number comes from a table in `results/`; every step has a figure. Commands are in the README.

## Headline

![headline](results/headline.png)

| # | Step added | RTF | Note |
|---|---|---|---|
| 1 | As shipped: torch bf16 + torch codec, 13 s reference, watermark | 10.6 | 2 utterances; earlier runs 17 |
| 2 | Q4 backbone + ONNX int8 codec | 2.27 | RAM 3.3 GB -> 1.3 GB |
| 3 | 5 s reference | 1.12 | prefill 1.39 -> 0.32 |
| 4 | watermark off | 1.11 | matters only in streaming |
| 5 | llama.cpp built for this CPU | 1.08 | +3 % |
| 6 | output rows sliced (our patch) | **0.87** | +20 %, audio byte-identical |
| 7 | decode threads pinned to P-cores | 0.87 | within noise |
| 8 | streaming, shipped settings | 2.56 | stalls up to 4.5 s |
| 9 | streaming, codec spin off, 50-frame chunks | **0.95** | no stalls, first audio 1.4 s |

- 12x end to end on the same weights; WER 2-6 % and speaker similarity 0.47-0.50 flat from row 2 on (`results/h1_table.md`).
- Three changes do almost all of it: quantised backbone + ONNX codec, shorter reference, output-row slice.
- One interleaved session, tuning voice. Unseen voices: batch 0.73, streaming 0.96, stall-free (test set below).

## Hardware and scope

Intel i7-1265U (2 P-cores + 8 E-cores, 12 threads, 15 W), 32 GB, no CUDA, Windows 11, "best performance" power mode on mains.

- The system under test is backbone + codec + phonemizer + watermark: the codec is a `NeuTTS` constructor argument and users hear waveforms, not tokens. Neuphonic's published numbers exclude it; ours report both.
- No weight changes anywhere. Retraining ideas (lower token rate, distilled decoder) are future work.
- Tuning used one voice (jo) and corpus lines 1-10; five other voices and the remaining lines were held out and run once at the end.

## Metrics

| Metric | Why |
|---|---|
| RTF, total and per stage (phonemize, prompt, prefill, generate, codec, watermark, glue) | where the time goes; glue ≈ 0 or a stage boundary is missing |
| decode tok/s, prefill tok/s | comparable with Neuphonic's table; prefill is a fixed per-utterance cost |
| peak RSS | memory ceiling |
| TTFA, worst stall | streaming: first sound, and whether playback ever runs dry |
| WER (Whisper small.en, compound-tolerant) | words intact |
| speaker similarity (ECAPA cosine vs the reference) | voice intact |
| seam distance (log-mel dB vs the batch waveform of the same seed) | chunking artefacts; sample-level SNR was wrong (phase, not audibility) |
| n_failed | utterances with no speech tokens |

## Measurement validity

| Decision | Reason |
|---|---|
| Preflight refuses to start above 20 % other-process CPU | the first sweep ran at 28-46 % and tokens/s fell 4x mid-run |
| Record and flag per-row interference; never gate, rescale or retry | tried a gate: the signal was a constant kernel/Defender offset, zero information, 4x run time |
| Paired seeds, interleaved rounds, fresh process per config | identical configs drift up to 25 % between sessions; compare only within a sweep |
| Microbench for levers, full harness to confirm | harness decode variance ±40 % run to run; microbench ±5 % |
| Power mode, AC state, preflight load recorded per run | "balanced" caps clocks |

## 1. Shipped options

![b1](results/b1_stages.png)

- Nothing is real time as shipped. bf16 is pathological on a CPU without bf16 instructions: prefill 40 tok/s vs 314 in fp32, same weights.
- Backbone is 85-90 % of the cost, prefill the larger half. ONNX int8 codec: 0.27 -> 0.10 RTF, 2.8 GB -> 1.0 GB. Watermark < 0.03 RTF. Distilled codec changes nothing: its decoder is the same 185.6M-parameter network, only the encoder was distilled.
- WER scorer must accept compound spellings ("drugstore" / "drug store"): 10 % -> 2 %. Details: `results/b1_table.md`.

## 2. Threads and reference length

![r1](results/r1_tradeoff.png)

- Prefill scales with threads (350 tok/s at 12 vs 296 at 4); decode is best on 4 threads pinned to P-cores (65 vs 47 at 12). E-cores are half a P-core on decode, a third on prefill. End to end the gain is 0-5 %: within noise (`results/tune_threads.md`, `g1_table.md`).
- Prompt is 940 tokens of which 653 are the 13 s reference: reference length is the lever. 6.4 s is free (similarity equal to the full clip); 5 s costs some voice identity; 3 s is real time but a different-sounding speaker (`results/r1_table.md`).

## 3. Why streaming was slower than batch

![s4](results/s4_streaming.png)

- Shipped streaming was 1.5-2.7x slower than batch for three reasons: the watermark per chunk (0.25-2.8 RTF), context re-decode (the vocoder is bidirectional with global attention, so chunks re-decode lookback frames and never converge to the batch output: `results/vocoder_receptive_field.json`), and generation itself running 2x slower.
- The generation slowdown was the **ONNX Runtime codec's worker threads spinning after each call** and starving the backbone: 35 steps/s after a codec call vs 95 steady. One session option (`allow_spinning=0`) restores it. Batch never showed it because it calls the codec once.
- Not the per-token Python path: a token-level loop (`experiments/stream.py`, token-identical to the reference) gave no gain and was dropped. It did find a reference bug: the tail chunk is placed one frame late.
- With spin off, chunk size trades codec cost against first audio; lookback trades seam quality against speed. Chosen: 50 / 50 (RTF 0.95, first audio 1.4 s, seams as shipped); low-latency 25 / 10 (first audio 0.9 s). Tables: `results/s1_table.md`, `s3_table.md`, `s4_table.md`.

## 4. The output projection slice

![slice](results/lm_head_slice.png)

| Fact | Number |
|---|---|
| vocabulary is text the model never emits | 194,246 rows: 65,536 speech + 262 special + 128,448 text |
| the projection is half of each decode step (hidden 576, 24 layers) | layers 117M MACs, projection 112M; speech rows only 38M |
| dropped rows carry nothing (17,498 steps, 100 utterances) | mass mean 3e-5, max 5e-3; 0 of 874,900 top-50 slots; best rank ever 335 |
| rows are independent dot products; nothing downstream reads them | kept scores bit-identical |
| a sampled text token would be silently dropped by the decode regex but stay in context | the slice removes that failure mode |

- Implemented as a 49-line llama.cpp patch (`llama_patch/llama_output_rows.patch`): a view of the row range on the quantised matrix, scattered into the full-size score buffer so samplers and bindings are untouched. Off by default; `LLAMA_OUTPUT_ROWS=start:count`.
- Tried first from outside (hidden vector + NumPy matmul): slower, because llama.cpp still runs its projection and float32 is 4x the memory traffic of its 8-bit matrix.
- Tests: kept-row scores identical, tokens identical for a seed on every voice tried, 40 of 40 WAVs byte-identical, decode +25-28 %, RTF -11-15 % (`experiments/slice_test.py`, `results/f1_table.md`).
- Build grid: the slice is worth +18 %, compiling for this CPU +7 %, and they stack; OpenMP beats llama.cpp's own pool; forcing threads to spin halves decode (`results/grid.md`).

## 5. Held-out test set

![test](results/test_voices.png)

- Five unseen voices x 20 unseen texts, batch and streaming, once, configuration frozen. Batch under real time on every voice (0.62-0.86); streaming 0.82-1.09 with **zero stalls in 99**, long passages included.
- Quality is uneven, and the cause is the clip, not the speed work: the two voices cut mid-sentence have 3-7x the WER. Rerun with their whole clips: emily 19 -> 11 %, dave 10.5 -> 3.9 %. **Recommendation: a short clip that ends on a sentence boundary**, not "5 s".
- One failure in 100: the model sampled its stop token first (voice steven, one seed). Reproduced on the unmodified wheel; the shipped code crashes on it in both modes; the harness records it.

## Final configuration

Q4 backbone on patched llama.cpp (native build, rows sliced) · ONNX int8 codec, spinning off · short reference ending on a sentence boundary · watermark off (product decision) · streaming 50-frame chunks, lookback 50 · optional 4 decode threads pinned to P-cores.

## Limits and rejected ideas

- Samples are small (6-20 texts x 2 seeds) to fit this laptop; the harness scales to the 70-text corpus and only final configurations got the larger run. Session drift means headline rows are only comparable within their run.
- Slice verified on Q4, English, top-50 sampling; threads and the build are specific to this chip.
- Rejected: playback-device codec (output is PCM); lower than 50 Hz tokens (retraining); training a decoder (GPU); core pinning in batch mode (stages are sequential); a gate on background load; a Python-side projection; llama.cpp batch sizes; KV-prefix reuse between utterances (real, ~26 % of prefill, but secondary and unmeasured).
- `docs/PLAN.md` is the original plan, kept as written.
