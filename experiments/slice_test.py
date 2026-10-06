"""Test the llama.cpp output-row patch (llama_output_rows.patch) against the same binary with slicing off.

    LLAMA_CPP_LIB_PATH=<abs path to llama_patch/lib-slice> uv run experiments/slice_test.py

1. kept-row scores identical, dropped rows -inf   2. decode speed, interleaved   3. token identity per seed
"""
import contextlib, ctypes, io, re, statistics as st, time
from pathlib import Path

import numpy as np
import torch

ROWS = "128261:65537"  # <|SPEECH_GENERATION_END|> + <|speech_0..65535|>, contiguous
_ucrt = ctypes.CDLL("ucrtbase")


def slicing(on: bool):
    _ucrt._putenv(f"LLAMA_OUTPUT_ROWS={ROWS if on else ''}".encode())  # the C runtime env the DLL reads


def main():
    from neutts import NeuTTS
    import neutts.neutts as _nn
    import llama_cpp

    slicing(False)
    with contextlib.redirect_stdout(io.StringIO()):
        tts = NeuTTS(backbone_repo="neuphonic/neutts-nano-q4-gguf", codec_repo="neuphonic/neucodec-onnx-decoder-int8")
    _nn.print = lambda *a, **k: None
    b = tts.backbone
    print("library:", llama_cpp.llama_cpp._lib._name)
    V = b.n_vocab()
    start, count = map(int, ROWS.split(":"))
    ref_codes = torch.load("samples/jo6.pt")
    ref_text = Path("samples/jo6.txt").read_text(encoding="utf-8").strip()
    texts = [l.strip() for l in Path("corpus.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    toks = [b.token_bos()] + b.tokenize(tts._ggml_prompt(ref_codes, ref_text, texts[0]).encode(), add_bos=False, special=True)
    logits = lambda: np.ctypeslib.as_array(b._ctx.get_logits(), shape=(V,)).copy()

    # 1. scores
    # llama.cpp reuses its compute graph when the batch shape repeats, so step once after the prompt:
    # the shape change forces a rebuild under the current setting.
    probe = start + 1
    slicing(False); b.reset(); b.eval(toks); b.eval([probe]); full = logits()
    slicing(True); b.reset(); b.eval(toks); b.eval([probe]); sl = logits()
    kept = slice(start, start + count)
    outside = np.r_[sl[:start], sl[start + count:]]
    print(f"1. kept rows: max |diff| {np.abs(full[kept] - sl[kept]).max():.2e}; dropped rows all -inf: {bool(np.isneginf(outside).all())}; slicing active: {bool(np.isneginf(sl[0]))}")

    # 2. speed, interleaved rounds (prefill + 150 decode steps each)
    res = {False: [], True: []}
    pre = {False: [], True: []}
    for rnd in range(6):
        for on in (False, True) if rnd % 2 == 0 else (True, False):
            slicing(on); b.reset()
            t = time.perf_counter(); b.eval(toks); pre[on].append(len(toks) / (time.perf_counter() - t))
            t = time.perf_counter()
            for k in range(150):
                b.eval([start + 1 + (k * 37) % 60000])
            res[on].append(150 / (time.perf_counter() - t))
    for on in (False, True):
        print(f"2. slicing {'on ' if on else 'off'}: decode {st.median(res[on]):6.1f} steps/s (min {min(res[on]):.0f}, max {max(res[on]):.0f})   prefill {st.median(pre[on]):5.0f} tok/s")
    print(f"   decode speed-up: {st.median(res[True]) / st.median(res[False]):.3f}x")

    # 3. token identity through the reference batch call (llama.cpp's own sampler + seed)
    same = total = 0
    gen = {False: [], True: []}
    for text in texts[:6]:
        for seed in (1000, 1001):
            out = {}
            for on in (False, True):
                slicing(on); tts._seed = seed
                t = time.perf_counter()
                out[on] = re.findall(r"<\|speech_(\d+)\|>", tts._infer_ggml(ref_codes, ref_text, text))
                gen[on].append(len(out[on]) / (time.perf_counter() - t))
            same += out[False] == out[True]; total += 1
    print(f"3. identical token sequences, slicing on vs off: {same}/{total}")
    print(f"   whole generate call incl. prefill + sampling: off {st.median(gen[False]):.1f} tok/s, on {st.median(gen[True]):.1f} tok/s ({st.median(gen[True]) / st.median(gen[False]):.3f}x)")
    slicing(False)


if __name__ == "__main__":
    main()
