"""Grid search over llama.cpp build variants x thread settings x thread wait policy, on the final configuration
(Q4, 5 s reference, sliced output). Backbone-only microbench, interleaved by round.

    uv run grid.py                 # -> results/grid.jsonl, results/grid.md
    uv run grid.py --table         # re-print the table

Build and wait policy are fixed per process (read at library load), so each (build, policy) pair is a worker
process; thread counts and pinning vary inside it. Three measurements per setting:
  prefill tok/s | decode steps/s, steady | decode steps/s with a codec call every 25 steps (what streaming does)
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# name -> (library folder, output slice on?). The two "unsliced" rows answer: slice vs CPU-optimised build, which is worth more?
BUILDS = {
    "generic, unsliced": ("lib-generic", False),   # closest to the prebuilt wheel
    "native, unsliced": ("lib-slice", False),
    "generic, sliced": ("lib-generic", True),
    "native, sliced": ("lib-slice", True),          # the final configuration's build
    "native, sliced, no-omp": ("lib-noomp", True),
    "native, sliced, lto": ("lib-lto", True),
}
POLICIES = ("active", "passive")            # OMP_WAIT_POLICY: spin vs sleep between parallel regions
THREADS = [                                  # (label, decode threads, prefill threads, decode pinned to P-cores)
    ("6/12", 6, 12, False), ("6/8", 6, 8, False), ("4/12 pinP", 4, 12, True), ("4/8 pinP", 4, 8, True), ("2/12 pinP", 2, 12, True),
]
ROWS = "128261:65537"
P_CORES = [0, 1, 2, 3]
OUT = ROOT / "results" / "grid.jsonl"


def worker(build, policy, rnd):
    import contextlib, io
    import numpy as np, psutil, torch
    from neutts import NeuTTS
    import neutts.neutts as _nn

    with contextlib.redirect_stdout(io.StringIO()):
        tts = NeuTTS(backbone_repo="neuphonic/neutts-nano-q4-gguf", codec_repo="neuphonic/neucodec-onnx-decoder-int8")
    _nn.print = lambda *a, **k: None
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # spinning codec workers starve the backbone; off in the final config
    tts.codec.session = ort.InferenceSession(tts.codec.session._model_path, so, providers=["CPUExecutionProvider"])
    b = tts.backbone
    proc = psutil.Process()
    proc.nice(psutil.HIGH_PRIORITY_CLASS)
    allc = list(range(psutil.cpu_count()))
    ref = torch.load("samples/jo5.pt")
    toks = [b.token_bos()] + b.tokenize(tts._ggml_prompt(ref, Path("samples/jo5.txt").read_text(encoding="utf-8").strip(), "The birch canoe slid on the smooth planks.").encode(), add_bos=False, special=True)
    codes = np.asarray(ref[:66], dtype=np.int32)[None, None, :]
    step_tok = lambda k: 128262 + (k * 37) % 60000
    b.eval(toks); [b.eval([step_tok(k)]) for k in range(10)]; tts.codec.decode_code(codes)  # warm-up

    with open(OUT, "a", encoding="utf-8") as f:
        for label, nt, nb, pin in THREADS:
            b._ctx.set_n_threads(nt, nb)
            proc.cpu_affinity(allc)
            b.reset()
            t = time.perf_counter(); b.eval(toks); prefill = len(toks) / (time.perf_counter() - t)
            if pin:
                proc.cpu_affinity(P_CORES)
            t = time.perf_counter()
            for k in range(120):
                b.eval([step_tok(k)])
            steady = 120 / (time.perf_counter() - t)
            spent = 0.0
            for k in range(100):
                if k % 25 == 0:                      # a streaming chunk boundary: the codec runs on all cores
                    proc.cpu_affinity(allc)
                    tts.codec.decode_code(codes)
                    if pin:
                        proc.cpu_affinity(P_CORES)
                t = time.perf_counter(); b.eval([step_tok(200 + k)]); spent += time.perf_counter() - t
            row = {"build": build, "policy": policy, "threads": label, "round": rnd, "prefill": round(prefill, 1), "decode": round(steady, 1), "decode_interrupted": round(100 / spent, 1)}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"r{rnd} {build:24s} {policy:8s} {label:10s} prefill {prefill:6.0f}  decode {steady:6.1f}  interrupted {100 / spent:6.1f}", flush=True)
    proc.cpu_affinity(allc)


def table():
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = sorted({(r["build"], r["policy"], r["threads"]) for r in rows})
    agg = []
    for k in keys:
        rs = [r for r in rows if (r["build"], r["policy"], r["threads"]) == k]
        agg.append((*k, st.median(r["prefill"] for r in rs), st.median(r["decode"] for r in rs), st.median(r["decode_interrupted"] for r in rs), len(rs)))
    agg.sort(key=lambda a: -a[5])
    md = ["| build | wait policy | threads dec/pre | prefill tok/s | decode steps/s | decode, interrupted | rounds |", "|---|---|---|---|---|---|---|"]
    md += [f"| {a[0]} | {a[1]} | {a[2]} | {a[3]:.0f} | {a[4]:.1f} | {a[5]:.1f} | {a[6]} |" for a in agg]
    (ROOT / "results" / "grid.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--worker", nargs=3)
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], a.worker[1], int(a.worker[2]))
    if a.table:
        return table()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("")
    # wait policy is only crossed on the final build; the others use the OpenMP default
    combos = [(bd, p) for bd in BUILDS for p in (POLICIES if bd == "native, sliced" else ("default",))]
    t0 = time.perf_counter()
    for rnd in range(a.rounds):
        for i, (bd, p) in enumerate(combos if rnd % 2 == 0 else combos[::-1]):
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONWARNINGS": "ignore", "LLAMA_CPP_LIB_PATH": str(ROOT / ("llama_patch" if BUILDS[bd][0] == "lib-slice" else "tools") / BUILDS[bd][0]), "LLAMA_OUTPUT_ROWS": ROWS if BUILDS[bd][1] else ""}
            if p in POLICIES:
                env["OMP_WAIT_POLICY"] = p
            done = rnd * len(combos) + i
            print(f"=== [{100 * done / (a.rounds * len(combos)):.0f}%] round {rnd + 1}/{a.rounds} {bd} / {p}  (elapsed {int(time.perf_counter() - t0) // 60}:{int(time.perf_counter() - t0) % 60:02d}) ===", flush=True)
            subprocess.run([sys.executable, __file__, "--worker", bd, p, str(rnd)], env=env, stderr=subprocess.DEVNULL)
    table()


if __name__ == "__main__":
    main()
