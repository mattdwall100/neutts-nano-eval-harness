"""Backbone-only microbench: prefill and decode tok/s vs thread count and core mask. No codec, no ASR.

Usage:
    uv run tune.py threads                    # -> results/tune_threads.jsonl + results/tune_threads.md
    uv run tune.py probe                      # 1-thread decode rate pinned to each logical CPU (finds P vs E cores) -> results/tune_probe.json

Interleaved: every round runs every setting once, so drift hits all settings equally.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import psutil
import torch

Q4 = "neuphonic/neutts-nano-q4-gguf"
ONNX8 = "neuphonic/neucodec-onnx-decoder-int8"
P_CORES = [0, 1, 2, 3]          # i7-1265U: 2 P-cores x 2 HT; verified by `probe` (47 vs 26 tok/s single-thread)
ALL = list(range(psutil.cpu_count()))
TEXT = "The source of the huge river is the clear spring, and the stray cat gave birth to kittens under the old porch."
NL = chr(10)


class Timed:
    """Wrap backbone.eval: first call of a generation (len>1) is prefill, the rest are decode steps.
    Optionally switches process affinity per phase (prefill_mask / decode_mask)."""

    def __init__(self, backbone):
        self.reset()
        self.prefill_mask = self.decode_mask = None
        orig = backbone.eval
        proc = psutil.Process()

        def ev(tokens, *a, **kw):
            if self.prefill_mask:
                proc.cpu_affinity(self.prefill_mask if len(tokens) > 1 else self.decode_mask)
            t0 = time.perf_counter()
            r = orig(tokens, *a, **kw)
            dt = time.perf_counter() - t0
            if len(tokens) > 1:
                self.prefill += dt
                self.prompt_tokens += len(tokens)
            else:
                self.decode += dt
                self.steps += 1
            return r

        backbone.eval = ev

    def reset(self):
        self.prefill = self.decode = 0.0
        self.prompt_tokens = self.steps = 0


def load():
    import contextlib, io
    from neutts import NeuTTS
    import neutts.neutts as _nn

    with contextlib.redirect_stdout(io.StringIO()):
        tts = NeuTTS(backbone_repo=Q4, codec_repo=ONNX8)
    _nn.print = lambda *a, **k: None
    ref_codes = torch.load("samples/jo.pt")
    ref_text = Path("samples/jo.txt").read_text(encoding="utf-8").strip()
    return tts.backbone, tts._ggml_prompt(ref_codes, ref_text, TEXT)


def one_sample(backbone, timed, prompt, n_threads, n_threads_batch, mask, max_tokens=120, decode_mask=None):
    psutil.Process().cpu_affinity(mask)
    timed.prefill_mask, timed.decode_mask = (mask, decode_mask) if decode_mask else (None, None)
    backbone._ctx.set_n_threads(n_threads, n_threads_batch)
    backbone.reset()
    timed.reset()
    t0 = time.perf_counter()
    backbone(prompt, max_tokens=max_tokens, temperature=1.0, top_k=50, stop=["<|SPEECH_GENERATION_END|>"], seed=1000)
    wall = time.perf_counter() - t0
    return {
        "prefill_tok_s": round(timed.prompt_tokens / timed.prefill, 1),
        "decode_tok_s": round(timed.steps / timed.decode, 1) if timed.steps else None,  # eval() only
        "loop_tok_s": round(timed.steps / (wall - timed.prefill), 1) if timed.steps else None,  # incl. sampling/detok
        "steps": timed.steps,
    }


# (label, n_threads (decode), n_threads_batch (prefill), process mask, decode-phase mask or None)
SETTINGS = [
    *[("P", t, t, P_CORES, None) for t in (1, 2, 3, 4)],
    *[("all", t, t, ALL, None) for t in (2, 4, 6, 8, 10, 12)],
    ("ref 6/12 all", 6, 12, ALL, None),        # reference code defaults on this machine
    ("4/10 all", 4, 10, ALL, None),             # best decode + best prefill counts, OS places threads
    ("4/10 dyn P|all", 4, 10, ALL, P_CORES),    # same, decode pinned to P-cores, prefill on all
    ("4/8 dyn P|all", 4, 8, ALL, P_CORES),
]


def table_from_jsonl():
    rows = [json.loads(l) for l in Path("results/tune_threads.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    write_table(rows)


def write_table(rows):
    md = ["| setting | threads dec/pre | prefill tok/s | decode tok/s (eval) | decode min-max | loop tok/s (incl. sampling) |", "|---|---|---|---|---|---|"]
    for label, t, tb, _, _ in SETTINGS:
        rs = [x for x in rows if x["setting"] == label and x["threads"] == t and x["threads_batch"] == tb]
        if not rs:
            continue
        d = [x["decode_tok_s"] for x in rs]
        md.append(f"| {label} | {t}/{tb} | {st.median(x['prefill_tok_s'] for x in rs):.0f} | {st.median(d):.1f} | {min(d):.0f}-{max(d):.0f} | {st.median(x['loop_tok_s'] for x in rs):.1f} |")
    print(NL + NL.join(md))
    Path("results/tune_threads.md").write_text(NL.join(md) + NL)


def run_threads(rounds):
    backbone, prompt = load()
    timed = Timed(backbone)
    one_sample(backbone, timed, prompt, 4, 4, P_CORES, max_tokens=20)  # warm-up
    rows = []
    with open("results/tune_threads.jsonl", "w", encoding="utf-8") as f:
        for r in range(rounds):
            for label, t, tb, mask, dmask in SETTINGS:
                s = one_sample(backbone, timed, prompt, t, tb, mask, decode_mask=dmask)
                row = {"round": r, "setting": label, "threads": t, "threads_batch": tb, **s}
                rows.append(row)
                f.write(json.dumps(row) + NL)
                f.flush()
                print(f"round {r} {label:16s} {t:2d}/{tb:2d}  prefill {s['prefill_tok_s']:6.1f}  decode {s['decode_tok_s']:6.1f}  loop {s['loop_tok_s']:6.1f} tok/s", flush=True)
    psutil.Process().cpu_affinity(ALL)
    write_table(rows)


def run_probe():
    backbone, prompt = load()
    timed = Timed(backbone)
    print("1-thread decode tok/s pinned to each logical CPU (P-cores should be ~2x E-cores):")
    rows = []
    for cpu in ALL:
        s = one_sample(backbone, timed, prompt, 1, 1, [cpu], max_tokens=40)
        rows.append({"cpu": cpu, **s})
        print(f"  cpu {cpu:2d}: decode {s['decode_tok_s']:5.1f}  prefill {s['prefill_tok_s']:5.1f}", flush=True)
    psutil.Process().cpu_affinity(ALL)
    Path("results/tune_probe.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["threads", "probe", "table"])
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS if sys.platform == "win32" else -10)
    {"probe": run_probe, "table": table_from_jsonl}.get(a.what, lambda: run_threads(a.rounds))()
