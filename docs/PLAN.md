# Plan: benchmark and optimise NeuTTS-Nano on-device

Working plan agreed so far. DESIGN.md will record what was actually done and why; this file records what we intend to do, in order of difficulty.

## Scope

**NeuTTS-Nano** means the shipped system: espeak-ng phonemizer, Llama backbone (24 layers, hidden 576, ~117M active params), NeuCodec decoder (50 Hz, single codebook, 24 kHz out), Perth watermark. The codec is in scope: it is a constructor argument of the reference `NeuTTS` class and the user hears waveforms, not tokens. Neuphonic's published numbers exclude it; ours will report both.

In scope: configuration, runtime, quantisation, threading, streaming parameters, exporting shipped weights to other runtimes.
Out of scope: training or fine-tuning any weights (backbone or codec). Named as future work only.

## Hardware

Intel i7-1265U (2 P-cores + 8 E-cores, 12 threads, 15 W), 32 GB RAM, Intel Iris Xe iGPU, no CUDA. Windows 11. Everything is CPU.

---

## Part A: Evaluation harness

### What "performance" means

Headline metrics, every configuration, every run:

| Metric | Definition |
|---|---|
| RTF | end-to-end wall time / seconds of audio produced. Below 1.0 is real time. |
| TTFA | time to first audio. Equals RTF-derived full latency in batch mode; measured per chunk in streaming mode. |
| Peak RSS | peak working set of the process, after load and after inference. |
| Backbone tok/s | tokens generated / generate-stage time. Directly comparable to Neuphonic's published table. |

Per-stage breakdown, stacked against end-to-end so hidden glue shows up:

1. phonemize
2. prompt build (tokenise, template)
3. token generation (backbone)
4. codec decode
5. watermark
6. glue = end-to-end minus sum of stages

Quality floor, so speed gains cannot hide quality loss:

- WER of generated audio via faster-whisper (base or small model, not tiny), against the input text.
- Speaker similarity between output and reference audio (speaker-embedding cosine).

### Statistical design

- Fixed text corpus of varied sentence lengths (`corpus.txt`).
- Each sentence repeated N times with seeds `seed_base + repeat`. The same seed set is used in every configuration, so comparisons are paired: a WER or RTF difference is attributable to the configuration, not to the sampling draw or the sentence.
- Report medians and spread, never a single run.

### Tiers

**A1 Easy (hello world)**
- `bench.py`: wraps the five stage methods of the reference class, records exclusive per-stage time, e2e, RTF, tok/s, peak RSS, seed. One JSONL row per utterance, WAVs saved, median summary. Done.
- `score.py`: WER and speaker similarity over a results directory.
- One command runs a named configuration.

**A2 Middle**
- Configuration matrix runner: one command sweeps backbone × codec × threads × watermark and writes a comparison table (markdown + CSV).
- Streaming-mode measurement: TTFA per chunk, chunk cadence, total RTF, seam quality (WER on stitched output vs batch output of the same seed).
- Warm vs cold: first-call latency separately from steady state.
- Hardware fingerprint recorded in every results file (CPU model, core counts, RAM, OS, Python, torch, llama.cpp versions).

**A3 Ambitious**
- Core-affinity aware runs: pin the process or stage thread pools to P-cores or E-cores and record which mask was used.
- Boundary-artefact metric for streaming (spectral discontinuity at chunk seams) in addition to WER.
- Power or energy per utterance if a Windows counter is accessible. Nice to have.
- Playback-device dimension was discussed and dropped: output is plain 24 kHz PCM, the OS mixer resamples to the device rate at negligible cost, and nothing in the pipeline depends on the sound card.

---

## Part B: Optimisation

### Levers available in the shipped system

| Stage | Levers |
|---|---|
| Backbone | torch fp32 vs bf16 (CPU has no native bf16: Alder Lake, AVX2 only), GGUF Q8, GGUF Q4, thread count, core mask, llama.cpp batch sizes, KV cache settings |
| Codec | neucodec (torch), distill-neucodec (torch), onnx-decoder fp32, onnx-decoder int8, **distilled decoder exported to ONNX int8 by us** (not shipped), ORT thread settings |
| Watermark | on / off (licence does not require it; shipping without it is a product decision) |
| Streaming | frames per chunk, lookback, lookforward, overlap-add power, per-chunk watermark cost |
| Token count | reference code forces `min_new_tokens=50`; inspect stop/max-length handling for wasted tokens |

Not available without retraining, documented as future work:
- Reducing the 50 Hz token rate. Baked into both backbone and codec training.
- Training a cheaper codec decoder by distillation. Conceptually easy (deterministic teacher, unlimited synthetic data) but needs a GPU and spectral/adversarial losses to sound clean. The shipped distilled decoder already captures most of this gain.

### Tiers

**B1 Easy**
- Baseline: torch bf16 backbone + torch neucodec + watermark, as the reference code ships.
- torch fp32 vs bf16 on the backbone.
- GGUF Q8 and Q4 backbones.
- ONNX fp32 and int8 codec decoders; distilled torch decoder.
- Watermark off.
- Expected outcome: Q4 + ONNX int8 + no watermark is the obvious floor. Everything after must beat it.

**B2 Middle**
- Thread count sweep for the backbone (2, 4, 6, 8, 10, 12) crossed with core mask (P-cores only vs all). llama.cpp's pool waits on its slowest thread, so E-cores may be stragglers.
- Separate thread settings for the codec stage.
- Export distilled decoder to ONNX, quantise to int8. Likely the fastest decode available.
- torch.compile or Intel-optimised ONNX Runtime / OpenVINO execution provider for the codec.
- Sampling and stop-condition tuning: `min_new_tokens`, EOS handling, context length, to stop generating wasted tokens.
- Streaming parameter sweep: minimise TTFA while holding total RTF and WER within tolerance of batch mode. Check whether per-chunk watermarking is a cost or a quality problem.

**B3 Ambitious**
- Pipeline overlap in streaming: backbone on P-cores, decode + watermark concurrently on E-cores. Only pays off when stages overlap; in batch mode the sum of stage times is what matters, so attack the largest term (backbone) first and keep that setting when partitioning.
- Export the backbone itself to ONNX or OpenVINO with int8, bypassing both torch and llama.cpp.
- Intel iGPU offload for the codec (OpenVINO GPU plugin) if the Iris Xe is faster than E-cores for the vocoder.
- Flat-RTF analysis: once pipelined, identify the bottleneck stage and rebalance cores so no stage starves the others.

### Order of work

1. B1 in full, measured with A1. Establishes the floor and the per-stage cost profile.
2. A2 matrix runner, then B2 using it. Thread/core results feed everything after.
3. Streaming (A2 streaming measurement, B2 sweep), then B3 overlap and affinity.
4. B3 backbone export and iGPU only if time allows and the profile says they would matter.

## Deliverables

- `README.md`: setup, one-command run, hardware.
- `DESIGN.md`: every decision and why, including the levers rejected and the future-work items above.
- `results/`: raw rows, summaries, comparison tables; checked in for the final configurations.
