# Design decisions

This is the story of the work, in the order it happened, with the reasoning I used at each step and the figure that settled it. Almost every number here comes from a table in `results/`; the few that come from one-off probes, such as the single-core rates and the spinning-thread measurement, are stated as such. The README has the commands.

## Where it ended up

![headline](results/headline.png)

| # | Step added | RTF | Note |
|---|---|---|---|
| 1 | As shipped: torch bf16 + torch codec, 13 s reference, watermark | 10.6 | 2 utterances; earlier runs of the same setup measured 17 |
| 2 | Q4 backbone + ONNX int8 codec | 2.27 | RAM 3.3 GB -> 1.0 GB |
| 3 | 5 s reference | 1.12 | prefill 1.39 -> 0.32 |
| 4 | watermark off | 1.11 | only matters in streaming |
| 5 | llama.cpp built for this CPU | 1.08 | +3 % |
| 6 | output rows sliced (my llama.cpp patch) | **0.87** | +20 %, audio byte-identical |
| 7 | decode threads pinned to P-cores | 0.87 | within noise |
| 8 | streaming with the shipped streaming settings, on row 7's pinned config | 2.56 | stalls up to 4.5 s; unpinned, the same settings measure 1.69 (`s3_table.md`): pinning and the codec's spinning threads interact badly |
| 9 | streaming, codec spinning off and 50-frame chunks (two changes; `s3_table.md` separates them) | **0.95** | no stalls, first audio after 1.4 s |

RTF is seconds of compute per second of audio, so below 1.0 is real time. Rows 1 to 7 are batch mode; 8 and 9 are streaming. Nothing is retrained: row 2 swaps in Neuphonic's published Q4 backbone and int8 codec, and rows 2 to 7 are the identical weights, a 2.6x gain between them. Word error rate stays at 2-6 % throughout; speaker similarity is flat in this run (0.47-0.50), though the dedicated reference sweep showed the 5 s clip costs a little (0.42 vs 0.49 at 6.4 s). For the slice in row 6 the output audio is byte for byte identical. The chain is one voice in one interleaved session; on five voices I never tuned on, with the clips as tested, the final configuration measures 0.73 in batch and 0.96 streaming, without a stall. One correction to the table in `h1_table.md`: its peak-RAM column reads 1.3 GB from row 2 on, which is a load-time artefact of the `gguf` Python package I had installed for the fork, since NeuTTS reads the model's metadata through a slow path when that package is present (+450 MB at load, 61 s instead of 18). The inference footprint is 1.0 GB, as in step 1, and the package is removed.

Three changes do almost all of it: the quantised backbone with the ONNX codec, the shorter reference clip, and the output-row slice. The rest of this document is how I found that out, including the things that didn't work.

## What I was measuring, and what I decided "performance" meant

The machine is an Intel i7-1265U laptop chip: two performance cores and eight efficiency cores, twelve threads, a 15 W power budget, 32 GB of RAM, no CUDA. Everything runs on the CPU, which is on-brief, since on-device is the point.

The first decision was what the system under test is. NeuTTS-Nano is four pieces: a phonemizer that turns text into phonemes, a Llama-style backbone that turns phonemes plus a reference voice into audio tokens, the NeuCodec decoder that turns those tokens into a 24 kHz waveform, and a Perth watermark applied to the result. Neuphonic's published numbers cover only the backbone's tokens per second. I decided the codec is in scope, because it is a constructor argument of their own `NeuTTS` class and because nobody hears tokens. So the harness reports both: backbone tokens per second, comparable to their table, and end-to-end real-time factor, which is what a user feels.

The second decision was a hard line: no weight changes. Quantisation choice, runtimes, threads, prompts and streaming settings are all fair game; training or fine-tuning anything is not. Two ideas that would have needed retraining, a lower token rate than 50 Hz and a cheaper distilled decoder, were parked as future work.

The metrics, then:

| Metric | Why |
|---|---|
| RTF, total and per stage: phonemize, prompt build, prefill, generate, codec decode, watermark, and "glue" | so I could see where the time goes. Glue is end-to-end minus the sum of the stages; if it grows, I've missed a stage boundary |
| decode tokens/s and prefill tokens/s | comparable with Neuphonic's table; prefill is a fixed cost per utterance that their table hides |
| peak memory | the on-device ceiling |
| time to first audio, and worst stall | for streaming: when the first sound arrives, and whether playback would ever run dry |
| word error rate, by transcribing the output with Whisper | are the words intact? |
| speaker similarity, cosine between ECAPA embeddings of the output and the reference clip | is the voice intact? |
| seam distance, the log-mel spectral difference between a streamed waveform and the batch waveform of the same seed | are the chunk joins audible? |

The quality metrics exist as a floor. Every speed trick I tried could have bought its speed by dropping words or drifting the voice, and I wanted the harness to catch that without me listening to hundreds of files.

## Building the harness, and learning what a trustworthy number costs

I wrapped Neuphonic's `NeuTTS` class rather than reimplementing it, so the numbers are for the code they ship. The stage timers are monkeypatched onto its methods, and because the torch and GGUF paths nest those methods differently, the timer subtracts nested intervals so nothing is counted twice. Every configuration uses the identical set of seeds, so when two configurations differ it's the configuration, not the dice. The corpus is seventy Harvard sentences, public domain and phonetically balanced, with no digits so WER needs no number handling.

The first real sweep taught me the most important lesson of the project. I ran it while Spotify, Chrome and Docker were open, and tokens per second on the same sentence fell from 1.7 to 0.4 over ten utterances. My first reaction was to measure the background load and rescale. That's wrong on a 15 W chip: another process doesn't just take a share of the CPU, it steals clock through the shared power budget, pollutes the cache and competes for memory bandwidth, and it can land on exactly the core you pinned to. There is no linear correction.

My second reaction was a gate: measure other-process CPU during each utterance, and discard and re-run anything above a threshold. That was also wrong, and it took a run to see why. On an idle machine every row still read 15 to 20 percent, 142 discards in twenty minutes, and the figure explained none of the variance in tokens per second. The "other" load was a constant accounting offset: kernel time paging the memory-mapped model, Defender scanning every WAV I wrote. A gate on a metric with no signal just burns time.

What I settled on is control and provenance rather than correction: a preflight that refuses to start if other processes are above 20 percent, naming them; per-row recording of that load with a "suspect" flag, never discarded; high process priority; a warm-up; and, for anything that matters, interleaved rounds, so that if the machine slows during a run every configuration slows together and the comparison stays paired. I also learned that identical configurations drift by up to 25 percent between sessions on this laptop, so the only comparisons I trust are rows measured in the same run. That is why the headline chain was re-measured in one session rather than stitched from the sweeps that discovered each step.

## Step 1: the shipped options

![b1](results/b1_stages.png)

Nothing is real time as shipped. The default torch path loads the backbone in bfloat16, and this CPU has no bfloat16 instructions, so the matrix multiplies fall to an emulation path: prefill ran at 40 tokens per second against 314 for the same weights in float32. The Q4 GGUF backbone through llama.cpp was the only sensible starting point, and the ONNX int8 codec decoder was the one clean win on the codec side, 0.27 down to 0.10 RTF and 2.8 GB down to 1.0 GB of memory with no measurable quality change.

Two smaller things fell out. The watermark costs under 0.03 RTF in batch, so it isn't a batch-mode lever at all. And the distilled codec Neuphonic ships changes nothing for decode speed, though it saves 0.4 GB of RAM, which puzzled me until I counted parameters: its decoder is the identical 185.6M-parameter network as the full codec, and only the encoder, which is used once per reference clip, was distilled. That closed the codec axis, and it also killed my idea of exporting the distilled decoder to ONNX: it would have reproduced the existing file.

The scorer needed a fix here too. Whisper writes "drugstore" where the corpus has "drug store", which a word-level alignment counts as two errors. Making the alignment accept a word matching two adjacent words on the other side took the GGUF configurations from a reported 10 percent WER to 2 percent, and the remaining errors were real misreadings.

## Step 2: where the backbone's time goes, and the reference clip

To tune the backbone I had to split its time into two phases, because they behave like different workloads. Prefill is reading the prompt: all of its tokens pushed through the transformer in one pass to build the attention cache. Decode is generating: one new token per pass, each one a full trip through the 24 layers plus the output projection, reusing the cache for everything before it. Per token, prefill is about eight times cheaper, because it pushes hundreds of tokens through each weight matrix at once and only needs output scores for the last one; decode has to go one token at a time.

That split exposed something the shipped code hides. The prompt is 940 tokens, and 653 of them are the 13-second reference clip. Prefill of that prompt was a constant 2.5 to 3.5 seconds per utterance on Q4, half the total for a two-second sentence, and it's redone from scratch every call because the reference code resets the model between utterances for reproducibility. The reference clip is the lever.

![r1](results/r1_tradeoff.png)

So I cut Neuphonic's sample voice at word boundaries, with Whisper timings aligned to the official transcript so the shortened text is exactly what's spoken, and swept six lengths from 3 to 13 seconds in one interleaved session. RTF is linear in prompt length. At 6.4 seconds the voice similarity was as good as the full clip; at 5 seconds it slipped slightly with occasional outliers; at 3 seconds the words were intact but it was a different-sounding speaker. I chose 5 seconds for the headroom and said so, with the cost recorded.

Threads were the other backbone lever and they turned out to be a small one. A microbench showed prefill likes all twelve threads, since the efficiency cores help a compute-bound pass, while decode is best on four threads pinned to the two performance cores, because llama.cpp's pool waits for its slowest thread and an efficiency core is half a performance core on this work. Pinning gave 13 percent in isolation and 0 to 5 percent end to end across three confirmation runs, which I count as noise. It's in the final configuration as optional and not in the claims.

## Step 3: why streaming was slower than batch, and the fix

Streaming is the configuration that matters for a product, and as shipped it was 1.6 times slower than batch at the default 25-frame chunks (1.73 against 1.06 in the first streaming sweep), up to 5 times at 5-frame chunks, and it stalled for seconds. Three things were going on, and it took two sweeps and a dead end to separate them.

The first was the watermark, applied per chunk in streaming with a fixed cost of a few hundred milliseconds per call: 0.25 RTF at the default chunk size and 2.8 at 5-frame chunks. It went off, with the note that shipping with it is Neuphonic's policy decision, and that the batch-mode cost of keeping it is under 0.03 RTF.

The second was the vocoder re-decoding context. The codec decoder is bidirectional: the audio it produces for a frame depends on frames after it, so each chunk is decoded with lookback frames of already-decoded context and a few lookforward frames, and only the middle is kept. I measured whether this converges by decoding a real token sequence with increasing lookback and comparing to the whole-sequence decode: it never plateaus, 4.2 dB at zero lookback down to 2.5 dB at 100, because the decoder has global attention over its window. So chunking has an inherent cost and lookback is a dial, not a threshold. Larger chunks amortise it.

The third was generation itself running twice as slowly in streaming as in batch, and my first hypothesis was wrong. I assumed the per-token Python overhead of the library's streaming completion path, so I wrote a token-level loop that drives the model with ids only, verified it produced identical tokens to the reference for every seed, and it gained nothing. It did find a small bug in the reference streaming code, the tail chunk placed one frame late, and it's kept in `experiments/` as the record of a wrong hypothesis.

The real cause only showed up when I built a microbench that interrupted decoding with a codec call every 25 steps, the way streaming does. In that probe, steady decode ran at 95 steps per second; after each codec call it ran at 35, and the first step after the call took 120 to 330 milliseconds instead of 11. ONNX Runtime's worker threads keep spinning after a call finishes, in case more work arrives, and in that window they occupy the cores the backbone needs. One session option, telling those workers to sleep when idle, restored generation to 85 to 99 steps per second. Batch mode never showed this because it calls the codec once per sentence.

![s4](results/s4_streaming.png)

With that fixed, chunk size and lookback became a clean two-dial trade, and I swept them. Every setting streamed without stalling. Lookback 50 gives the best seams and voice match; shorter lookback buys speed and earlier first audio. I chose 50-frame chunks with lookback 50 as the default, because its seam quality equals the shipped setting and it has the most headroom, and 25-frame chunks with lookback 10 as a low-latency option, first sound after 0.9 seconds instead of 1.4.

## Step 4: the output projection, and forking llama.cpp

With prefill cut down, generation was the largest stage, so I looked at what one decode step actually computes. The transformer's last layer produces a 576-dimensional vector, and the final step multiplies it against one row per vocabulary entry to score every possible next token. NeuTTS-Nano kept the 194,256-entry vocabulary of the text model it was fine-tuned from, and with a hidden size of 576 that one matrix is about as much work as the 24 layers combined: 112 million multiply-adds against 117 million.

| Fact | Number |
|---|---|
| the vocabulary is mostly text the model never emits | 194,256 rows: 65,536 speech codes + 262 special tokens + 128,448 text tokens + 10 unused |
| the projection is half of each decode step | layers 117M MACs, projection 112M; the speech rows alone would be 38M |
| each row's score is an independent dot product; nothing downstream reads the rows we'd drop | kept scores are bit-identical |

The question was whether the text rows ever matter, and that is measurable: at every decode step, how much probability do they hold, and do any reach the top-50 candidate set the sampler draws from? Over 100 utterances and 17,498 decode steps the dropped rows held a mean of 0.00003 of the probability, never once appeared in the top 50, and the closest any came was rank 335. The rows that came closest were phoneme characters from the prompt, not English words.

![slice](results/lm_head_slice.png)

There's a second reason to want the slice. Today nothing forbids the model from emitting a text token; it just doesn't, as a learned habit. If it did, the reference code's regex would silently drop that frame, but the token would stay in the backbone's context and condition everything after it. Slicing makes that impossible by construction.

The first implementation failed, and the failure taught me where the saving had to live. I pulled the hidden vector out of llama.cpp and did the sliced projection in NumPy: the scores matched to a correlation of 0.999998, but the step got slower, because llama.cpp still ran its full projection in that mode, and my float32 matrix was four times the memory traffic of its 8-bit one. The slice had to happen inside llama.cpp, in its own format.

There was no compiler on the machine and no admin rights, so I unzipped a portable LLVM-MinGW toolchain, checked out llama.cpp at the exact revision the Python bindings vendor, and wrote a 49-line patch: a view of the row range on the quantised output matrix, with the scores scattered into the usual full-size buffer and minus infinity elsewhere, so the sampler, the bindings and the whole harness stay untouched. It's off by default and switched on by an environment variable, which means the same binary is its own baseline. The tests: kept-row scores identical, token sequences identical for a seed on the voice I tested it with (jo at two clip lengths, 12 of 12 cases), 40 of 40 output files byte-identical, decode 25 to 28 percent faster, end-to-end RTF 11 to 15 percent lower. Byte-identical audio means there is no quality question to answer.

A build grid separated the two things that had changed together when I forked: at the library's default six threads, compiling for this CPU is worth about 7 percent and the slice about 18, and they stack; at four pinned threads the slice is worth 33 percent and the build nothing measurable. It also showed that forcing llama.cpp's own threads to spin halves decode, the mirror image of the codec finding.

## Step 5: the held-out test

![test](results/test_voices.png)

Everything above was tuned on one voice and ten sentences, which is a validation set by any other name. So the last measurement was a test set: five voices I had never used, twenty texts I had never tuned on including the long passages, batch and streaming, once on the frozen configuration. It was interrupted twice by the model failure described below, which crashed the shipped code; I patched the harness to record that case and resumed, with nothing in the configuration changed. One qualification on "never used": the projection-slice measurement had run corpus lines 1-40 and 61-70 through the model on the tuning voice to count probabilities, which covers the test texts. No decision was tuned on them, but they were not entirely unseen by the model on that voice. Lines 26-60 and 66-70 were never used at all.

Speed generalised. Batch was under real time on every voice, 0.62 to 0.86; streaming ran 0.82 to 1.09 by voice with zero stalls in 99 utterances, long passages included.

Quality did not generalise uniformly, and the test set did its job. Two voices had three to seven times the word error rate of the others, and they were the two whose clips I had cut to five seconds mid-sentence to match the tuning voice. Rerunning them with their whole clips roughly halved the errors: 19 to 11 percent, 10.5 to 3.9. The typical failure is a repeated or dropped phrase. I should be careful about the cause, though. The whole clips are also longer, 8.1 and 7.5 seconds, so the rerun changed length and ending together; and the tuning clip itself ends on a comma mid-sentence without showing the problem. What I can say is that these two voices need more of their clip than five seconds, and that the whole-clip reruns are slower, RTF 0.95 and 1.23, so "under real time on every voice" holds for the clips as tested and not for every clip length. The honest recommendation is a short clip ending on a natural pause, checked per voice, and the harness is how to check.

The test set also produced one failed utterance in a hundred: the model sampled its stop token first and produced nothing. I reproduced it on the unmodified library with the same seed, so it isn't mine, and I found that the shipped code crashes on it in both batch and streaming. The harness now records it as a failure and carries on, because a benchmark that crashes on the model's own failure can't report the failure rate.

## The final configuration, and what I'd tell the team

Q4 backbone on llama.cpp built for this CPU with the output rows sliced; ONNX int8 codec with thread spinning off; a short reference clip ending on a natural pause, checked per voice; watermark off as a product decision; streaming with 50-frame chunks and lookback 50, or 25 and 10 for low latency; four decode threads pinned to the performance cores as an optional extra.

The two findings I'd put in front of Neuphonic are the ones that cost nothing and apply everywhere: the output-row slice, a few lines in their own llama.cpp build or export script with the evidence above, and the codec's spinning threads, a one-line session option that is the entire reason their streaming mode is slower than their batch mode.

## Limits, and ideas I rejected

The samples are small, six to twenty sentences with two seeds, because that is what twenty-minute sweeps on a laptop allow. The harness scales to the full corpus, and only final configurations got the larger run. Thread settings and the compiled library are specific to this chip; the harness, the slice and the streaming fixes are not. The slice is verified on Q4, English and top-50 sampling.

Rejected along the way: optimising for the playback device, since the output is plain PCM and the OS resamples it for free; a lower token rate, which is baked into both models; training a cheaper decoder, which needs a GPU; pinning different pipeline stages to different cores in batch mode, where stages run one after another so the sum is what matters (pinning the backbone's own decode threads is a different thing, and is in the final configuration as optional); the load gate; the Python-side projection; llama.cpp batch sizes; and reusing the cached prompt prefix between utterances, which is real, about a quarter of prefill, but secondary, and I left it unmeasured. `docs/PLAN.md` is the plan I wrote at the start, kept as written.
