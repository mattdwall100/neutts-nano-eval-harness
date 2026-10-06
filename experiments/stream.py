"""Token-level streaming for NeuTTS GGUF backbones.

Replaces the reference `infer_stream`: drives llama.cpp's generate() directly (token ids only; no detokenise,
stop-string scan or response dict per token), supports a growing chunk schedule, optional watermark, and
optional reuse of the KV cache for the shared prompt prefix. Same sampler chain and seed as the reference
batch call, so a seed yields identical tokens (verified by `uv run stream.py check`).

    for wav_chunk in stream(tts, text, ref_codes, ref_text, seed, chunk_schedule=(10, 25, 50, 100), lookback=25): ...
"""
import numpy as np

SPEECH_END = "<|SPEECH_GENERATION_END|>"


def _ids(backbone):
    s0 = backbone.tokenize(b"<|speech_0|>", add_bos=False, special=True)[0]
    end = backbone.tokenize(SPEECH_END.encode(), add_bos=False, special=True)[0]
    return s0, end


def generate_codes(tts, text, ref_codes, ref_text, seed, temperature=1.0, top_k=50, reuse_prefix=False):
    """Yield speech code ints one at a time, sampled exactly as the reference batch call samples them."""
    b = tts.backbone
    s0, end_id = _ids(b)
    # create_completion prepends BOS (prompt is 940 tokens, not 939); replicate so sampling is bit-identical
    toks = [b.token_bos()] + b.tokenize(tts._ggml_prompt(ref_codes, ref_text, text).encode(), add_bos=False, special=True)
    if not reuse_prefix:
        b.reset()  # reference behaviour: full re-prefill. Otherwise generate() reuses the longest cached prefix.
    b._seed = seed
    budget = tts.max_context - len(toks)
    # create_completion defaults the reference does not override: top_p .95, min_p .05, typical 1, repeat_penalty 1
    for n, tok in enumerate(b.generate(toks, top_k=top_k, top_p=0.95, min_p=0.05, typical_p=1.0, temp=temperature, repeat_penalty=1.0, reset=True)):
        if tok == end_id or n >= budget:
            return
        yield tok - s0


def _ola(chunks, offsets, power=1.0):
    """Linear overlap-add with explicit per-chunk sample offsets (reference impl assumes one stride)."""
    total = max(o + len(c) for c, o in zip(chunks, offsets))
    out = np.zeros(total, np.float32)
    wsum = np.zeros(total, np.float32)
    for c, o in zip(chunks, offsets):
        t = np.linspace(0, 1, len(c) + 2, dtype=np.float32)[1:-1]
        w = (0.5 - np.abs(t - 0.5)) ** power
        out[o : o + len(c)] += w * c
        wsum[o : o + len(c)] += w
    return out / np.maximum(wsum, 1e-8)


def stream(tts, text, ref_codes, ref_text, seed, chunk_schedule=(25,), lookback=50, lookforward=5, overlap=1,
           watermark=True, reuse_prefix=False, temperature=1.0, top_k=50):
    """Yield float32 audio chunks at 24 kHz. Chunk size follows `chunk_schedule` (last value repeats).

    Per chunk the vocoder decodes [lookback | chunk | lookforward] (+overlap each side) frames and keeps the chunk
    plus 2*overlap frames, exactly as the reference; chunks are blended by linear overlap-add. The tail is placed
    at its true offset (the reference places it one chunk stride on, which lengthens the output by `overlap` frames).
    """
    hop = tts.hop_length
    ref = [int(c) for c in (ref_codes.tolist() if hasattr(ref_codes, "tolist") else ref_codes)]
    codes = list(ref)
    n_ref = len(ref)
    done = n_ref  # frames (incl. reference) whose audio has been committed
    chunks, offsets, emitted, k = [], [], 0, 0

    def decode(seq):
        return tts._decode("".join(f"<|speech_{c}|>" for c in seq))  # via tts._decode so the harness stage wrapper sees it

    def wm(x):
        return x if (not watermark or tts.watermarker is None) else tts.watermarker.apply_watermark(x, sample_rate=24_000)

    for code in generate_codes(tts, text, ref_codes, ref_text, seed, temperature, top_k, reuse_prefix):
        codes.append(code)
        C = chunk_schedule[min(k, len(chunk_schedule) - 1)]
        if len(codes) - done < C + lookforward:
            continue
        a = max(done - lookback - overlap, 0)
        recon = wm(decode(codes[a : done + C + lookforward + overlap]))
        s = (done - a) * hop
        chunks.append(recon[s : s + (C + 2 * overlap) * hop])
        offsets.append((done - n_ref) * hop)
        done += C
        k += 1
        out = _ola(chunks, offsets)
        committed = (done - n_ref) * hop  # audio beyond this is still provisional (overlap region)
        yield out[emitted:committed]
        emitted = committed

    rem = len(codes) - done
    if rem > 0:
        a = max(done - lookback - overlap, 0)
        recon = wm(decode(codes[a:]))
        s = (done - a - overlap) * hop
        chunks.append(recon[s:])
        offsets.append((done - n_ref - overlap) * hop)
    if chunks:
        out = _ola(chunks, offsets)
        yield out[emitted:]


# ---------------------------------------------------------------- self-check: token identity with the reference
def check(n_texts=6, seeds=(1000, 1001)):
    import contextlib, io, re
    from pathlib import Path

    import torch
    from neutts import NeuTTS
    import neutts.neutts as _nn

    with contextlib.redirect_stdout(io.StringIO()):
        tts = NeuTTS(backbone_repo="neuphonic/neutts-nano-q4-gguf", codec_repo="neuphonic/neucodec-onnx-decoder-int8")
    _nn.print = lambda *a, **k: None
    ref_codes = torch.load("samples/jo8.pt")
    ref_text = Path("samples/jo8.txt").read_text(encoding="utf-8").strip()
    texts = [l.strip() for l in Path("corpus.txt").read_text(encoding="utf-8").splitlines() if l.strip()][:n_texts]
    same = {"reset": 0, "reuse": 0}
    total = 0
    for text in texts:
        for seed in seeds:
            tts._seed = seed
            ref_out = tts._infer_ggml(ref_codes, ref_text, text)
            ref_ids = [int(x) for x in re.findall(r"<\|speech_(\d+)\|>", ref_out)]
            ours = list(generate_codes(tts, text, ref_codes, ref_text, seed))
            reuse = list(generate_codes(tts, text, ref_codes, ref_text, seed, reuse_prefix=True))
            same["reset"] += ours == ref_ids
            same["reuse"] += reuse == ref_ids
            total += 1
            print(f"seed {seed} | {text[:38]:38s} | ref {len(ref_ids):3d} tok | ours {len(ours):3d} {'==' if ours == ref_ids else '!='} | prefix-reuse {len(reuse):3d} {'==' if reuse == ref_ids else '!='}", flush=True)
    print(f"identical to reference: full reset {same['reset']}/{total}, prefix reuse {same['reuse']}/{total}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
