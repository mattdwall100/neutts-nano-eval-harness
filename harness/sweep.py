"""Run a named set of bench.py configurations, each in a fresh process, score them, print a comparison table.

Usage:
    uv run sweep.py b1                      # the sweep's own limit/repeats (sized for ~20 min idle)
    uv run sweep.py b1 --interleave         # round-robin repeats across configs (final numbers)
    uv run sweep.py --table results/b1_*    # just re-print the table

Each config is a dict of bench.py CLI args; 'limit'/'repeats' keys override the sweep defaults.
A fresh process per config keeps peak-RSS honest. Scoring runs once at the end in one process.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

NANO, Q8, Q4 = "neuphonic/neutts-nano", "neuphonic/neutts-nano-q8-gguf", "neuphonic/neutts-nano-q4-gguf"
CODEC, DISTILL, ONNX, ONNX8 = (
    "neuphonic/neucodec",
    "neuphonic/distill-neucodec",
    "neuphonic/neucodec-onnx-decoder",
    "neuphonic/neucodec-onnx-decoder-int8",
)

ROOT = Path(__file__).resolve().parent.parent  # repo root: run from anywhere
HERE = Path(__file__).resolve().parent
LIB_SLICE = str(ROOT / "llama_patch" / "lib-slice")  # llama.cpp built with llama_output_rows.patch
SPEECH_ROWS = "128261:65537"  # <|SPEECH_GENERATION_END|> + the 65,536 speech codes, contiguous

JO5 = dict(ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt")
PIN = dict(threads=4, threads_batch=12, decode_cores="0,1,2,3")
FINAL_ENV = {"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS}

SWEEPS = {
    # B1: shipped options only. Backbone axis with the stock codec, codec axis with the Q4 backbone.
    # Stage timing is separable, and decode is deterministic given tokens, so a full cross is unnecessary.
    # Sized to ~20 min on an idle i7-1265U: torch backbones run at RTF 10-30, 2 utterances establish "unusable".
    "b1": {
        "_defaults": dict(limit=6, repeats=2),
        "b1_torch_bf16": dict(backbone=NANO, codec=CODEC, dtype="bf16", limit=2, repeats=1),
        "b1_torch_fp32": dict(backbone=NANO, codec=CODEC, dtype="fp32", limit=2, repeats=1),
        "b1_q8": dict(backbone=Q8, codec=CODEC),
        "b1_q4": dict(backbone=Q4, codec=CODEC),
        "b1_q4_distill": dict(backbone=Q4, codec=DISTILL),
        "b1_q4_onnx": dict(backbone=Q4, codec=ONNX),
        "b1_q4_onnx8": dict(backbone=Q4, codec=ONNX8),
        "b1_q4_onnx8_nowm": dict(backbone=Q4, codec=ONNX8, no_watermark=True),
    },
    # B2: tune the B1 winner (Q4 + ONNX int8). Same texts/seeds as B1 so rows are comparable to b1_q4_onnx8.
    # Speaker similarity is always scored against the full 13 s clip so shorter references are judged fairly.
    "b2": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, sim_ref="samples/jo.wav"),
        "b2_base": dict(),
        "b2_threads": dict(threads=4, threads_batch=10, decode_cores="0,1,2,3"),
        "b2_ref8": dict(ref_audio="samples/jo8.wav", ref_text="samples/jo8.txt"),
        "b2_ref5": dict(ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt"),
        "b2_threads_ref5": dict(threads=4, threads_batch=10, decode_cores="0,1,2,3", ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt"),
    },
    # R1: reference length, full range in one session, interleaved. Speaker sim always vs the full 13 s clip.
    "r1": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, sim_ref="samples/jo.wav"),
        "r1_ref3": dict(ref_audio="samples/jo3.wav", ref_text="samples/jo3.txt"),
        "r1_ref5": dict(ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt"),
        "r1_ref6": dict(ref_audio="samples/jo6.wav", ref_text="samples/jo6.txt"),
        "r1_ref8": dict(ref_audio="samples/jo8.wav", ref_text="samples/jo8.txt"),
        "r1_ref10": dict(ref_audio="samples/jo10.wav", ref_text="samples/jo10.txt"),
        "r1_ref13": dict(ref_audio="samples/jo.wav", ref_text="samples/jo.txt"),
    },
    # S2: streaming ladder, one change per row. Base = Q4 + ONNX int8 + 8.5 s reference (swap via _defaults after r1).
    "s2": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, ref_audio="samples/jo8.wav", ref_text="samples/jo8.txt", sim_ref="samples/jo.wav"),
        "s2_batch": dict(),
        "s2_ref_stream": dict(stream=True, chunk_frames=25, lookforward=5, lookback=50),
        "s2_tok": dict(stream=True, stream_impl="ours", chunk_frames=25, lookforward=5, lookback=50),
        "s2_tok_lb25": dict(stream=True, stream_impl="ours", chunk_frames=25, lookforward=5, lookback=25),
        "s2_tok_grow": dict(stream=True, stream_impl="ours", chunk_schedule="10,25,50,100", lookforward=5, lookback=25),
        "s2_tok_grow_nowm": dict(stream=True, stream_impl="ours", chunk_schedule="10,25,50,100", lookforward=5, lookback=25, no_watermark=True),
        "s2_tok_grow_nowm_pin": dict(stream=True, stream_impl="ours", chunk_schedule="10,25,50,100", lookforward=5, lookback=25, no_watermark=True, threads=4, threads_batch=10, decode_cores="0,1,2,3"),
    },
    # F1: final step. Same patched llama.cpp binary (llama_patch/lib-slice) for every row; only LLAMA_OUTPUT_ROWS differs.
    "f1": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=10, repeats=2, no_watermark=True, sim_ref="samples/jo.wav"),
        "f1_ref6_full": dict(ref_audio="samples/jo6.wav", ref_text="samples/jo6.txt", env={"LLAMA_CPP_LIB_PATH": LIB_SLICE}),
        "f1_ref6_sliced": dict(ref_audio="samples/jo6.wav", ref_text="samples/jo6.txt", env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS}),
        "f1_ref5_full": dict(ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt", env={"LLAMA_CPP_LIB_PATH": LIB_SLICE}),
        "f1_ref5_sliced": dict(ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt", env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS}),
    },
    # V1: voices. Final pipeline (Q4 + ONNX int8, no watermark, patched llama.cpp) on six speakers at native clip
    # length, 5 s cuts of three of them, and unsliced twins for two (byte-identity + speed-up beyond one voice).
    # Speaker similarity is scored against each voice's own full clip.
    "v1": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, no_watermark=True),
        **{f"v1_{v}_sliced": dict(ref_audio=f"samples/{v}.wav", ref_text=f"samples/{v}.txt", sim_ref=f"samples/{v}.wav", env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS})
           for v in ("jo", "emily", "dave", "paul", "steven", "sophie")},
        **{f"v1_{v}_full": dict(ref_audio=f"samples/{v}.wav", ref_text=f"samples/{v}.txt", sim_ref=f"samples/{v}.wav", env={"LLAMA_CPP_LIB_PATH": LIB_SLICE})
           for v in ("emily", "dave")},
        **{f"v1_{v}5_sliced": dict(ref_audio=f"samples/{v}5.wav", ref_text=f"samples/{v}5.txt", sim_ref=f"samples/{v}.wav", env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS})
           for v in ("jo", "emily", "dave")},
    },
    # T1: thread count / P-core pinning on the final configuration (5 s reference, sliced, no watermark).
    "t1": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, no_watermark=True, ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt", sim_ref="samples/jo.wav",
                          env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS}),
        "t1_default_6_12": dict(),
        "t1_4_12": dict(threads=4, threads_batch=12),
        "t1_4_12_pinP": dict(threads=4, threads_batch=12, decode_cores="0,1,2,3"),
        "t1_4_10_pinP": dict(threads=4, threads_batch=10, decode_cores="0,1,2,3"),
        "t1_2_12_pinP": dict(threads=2, threads_batch=12, decode_cores="0,2"),
        "t1_8_12": dict(threads=8, threads_batch=12),
    },
    # S3: final configuration, batch vs streaming at fixed chunk sizes (shipped streaming loop, watermark off).
    "s3": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, no_watermark=True, ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt", sim_ref="samples/jo.wav",
                          env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS}),
        "s3_batch": dict(),
        "s3_batch_nospin": dict(codec_no_spin=True),
        "s3_stream_25": dict(stream=True, chunk_frames=25, lookforward=5, lookback=50),                       # shipped streaming settings
        "s3_stream_25_nospin": dict(stream=True, chunk_frames=25, lookforward=5, lookback=50, codec_no_spin=True),
        "s3_stream_10_nospin": dict(stream=True, chunk_frames=10, lookforward=5, lookback=50, codec_no_spin=True),
        "s3_stream_50_nospin": dict(stream=True, chunk_frames=50, lookforward=5, lookback=50, codec_no_spin=True),
        "s3_stream_100_nospin": dict(stream=True, chunk_frames=100, lookforward=5, lookback=50, codec_no_spin=True),
        "s3_stream_25_lb25_nospin": dict(stream=True, chunk_frames=25, lookforward=5, lookback=25, codec_no_spin=True),
    },
    # S4: final streaming grid, chunk size x vocoder lookback, codec spinning off. Batch twin = seam reference.
    "s4": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, no_watermark=True, codec_no_spin=True, ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt", sim_ref="samples/jo.wav",
                          env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS}),
        "s4_batch": dict(),
        **{f"s4_c{c}_lb{lb}": dict(stream=True, chunk_frames=c, lookforward=5, lookback=lb) for c in (25, 50) for lb in (10, 25, 50)},
    },
    # G1: confirm the build-grid winners end to end (batch + default streaming), final configuration otherwise.
    "g1": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, no_watermark=True, codec_no_spin=True, ref_audio="samples/jo5.wav", ref_text="samples/jo5.txt", sim_ref="samples/jo.wav"),
        **{f"g1_{mode}_{name}": dict(**extra, **(dict(stream=True, chunk_frames=50, lookforward=5, lookback=50) if mode == "stream" else {}),
                                     env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS, **env})
           for mode in ("batch", "stream")
           for name, extra, env in (
               ("current", {}, {}),
               ("passive", {}, {"OMP_WAIT_POLICY": "passive"}),
               ("passive_pin4", dict(threads=4, threads_batch=12, decode_cores="0,1,2,3"), {"OMP_WAIT_POLICY": "passive"}),
           )},
    },
    # TEST: held-out test set, run once on the frozen final configuration. Voices and corpus lines never used in tuning
    # (tuning used voice "jo" and lines 1-10). Speaker similarity vs each voice's own full clip.
    "test": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, lines="11-25,61-65", repeats=1, no_watermark=True, codec_no_spin=True,
                          threads=4, threads_batch=12, decode_cores="0,1,2,3", env={"LLAMA_CPP_LIB_PATH": LIB_SLICE, "LLAMA_OUTPUT_ROWS": SPEECH_ROWS}),
        **{f"test_{mode}_{v}": dict(ref_audio=f"samples/{clip}.wav", ref_text=f"samples/{clip}.txt", sim_ref=f"samples/{v}.wav",
                                    **(dict(stream=True, chunk_frames=50, lookforward=5, lookback=50) if mode == "stream" else {}))
           for mode in ("batch", "stream")
           for v, clip in (("emily", "emily5"), ("dave", "dave5"), ("paul", "paul"), ("steven", "steven"), ("sophie", "sophie"))},
    },
    # QUICK: the one-command check. Shipped Q4 + ONNX codec vs the final configuration, 6 texts x 2 seeds, ~5 min.
    "quick": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, sim_ref="samples/jo.wav"),
        "quick_shipped": dict(),
        "quick_final_batch": dict(**JO5, no_watermark=True, **PIN, env=FINAL_ENV),
        "quick_final_stream": dict(**JO5, no_watermark=True, **PIN, env=FINAL_ENV, stream=True, chunk_frames=50, lookforward=5, lookback=50, codec_no_spin=True),
    },
    # H1: headline chain, one interleaved session, one voice (jo: the only one with a 13 s clip). Each row adds one change.
    "h1": {
        "_defaults": dict(lines="11-20", repeats=2, sim_ref="samples/jo.wav"),
        "h1_1_shipped": dict(backbone=NANO, codec=CODEC, dtype="bf16", lines="11-12", repeats=1),
        "h1_2_q4_onnx8": dict(backbone=Q4, codec=ONNX8),
        "h1_3_ref5": dict(backbone=Q4, codec=ONNX8, **JO5),
        "h1_4_nowm": dict(backbone=Q4, codec=ONNX8, **JO5, no_watermark=True),
        "h1_5_native_build": dict(backbone=Q4, codec=ONNX8, **JO5, no_watermark=True, env={"LLAMA_CPP_LIB_PATH": LIB_SLICE}),
        "h1_6_sliced": dict(backbone=Q4, codec=ONNX8, **JO5, no_watermark=True, env=FINAL_ENV),
        "h1_7_pinned": dict(backbone=Q4, codec=ONNX8, **JO5, no_watermark=True, **PIN, env=FINAL_ENV),
        "h1_8_stream_shipped": dict(backbone=Q4, codec=ONNX8, **JO5, no_watermark=True, **PIN, env=FINAL_ENV, stream=True, chunk_frames=25, lookforward=5, lookback=50),
        "h1_9_stream_final": dict(backbone=Q4, codec=ONNX8, **JO5, no_watermark=True, **PIN, env=FINAL_ENV, stream=True, chunk_frames=50, lookforward=5, lookback=50, codec_no_spin=True),
    },
    # S1: streaming on the B2 winner (Q4 + ONNX int8 + 8.5 s reference). s1_batch is the seam reference (same seeds -> same tokens).
    "s1": {
        "_defaults": dict(backbone=Q4, codec=ONNX8, limit=6, repeats=2, ref_audio="samples/jo8.wav", ref_text="samples/jo8.txt", sim_ref="samples/jo.wav"),
        "s1_batch": dict(),
        "s1_stream_25_5_50": dict(stream=True, chunk_frames=25, lookforward=5, lookback=50),   # reference defaults
        "s1_stream_10_5_50": dict(stream=True, chunk_frames=10, lookforward=5, lookback=50),
        "s1_stream_10_5_25": dict(stream=True, chunk_frames=10, lookforward=5, lookback=25),
        "s1_stream_10_2_25": dict(stream=True, chunk_frames=10, lookforward=2, lookback=25),
        "s1_stream_5_2_25": dict(stream=True, chunk_frames=5, lookforward=2, lookback=25),
    },
}
SEAM_REF = {"s1": "s1_batch", "s2": "s2_batch", "s3": "s3_batch", "s4": "s4_batch"}

COLS = [
    ("rtf_median", "RTF"),
    ("rtf_p25", "RTF p25"),
    ("rtf_p75", "RTF p75"),
    ("e2e_s_median", "e2e s"),
    ("ttfa_median", "TTFA s"),
    ("stall_max", "stall max s"),
    ("n_chunks_median", "chunks"),
    ("seam_logmel_db_median", "seam mel dB"),
    ("tok_per_s_median", "decode tok/s"),
    ("prefill_tok_per_s_median", "prefill tok/s"),
    ("prompt_tokens_median", "prompt tok"),
    ("stage_rtf_median.prefill", "prefill RTF"),
    ("stage_rtf_median.generate", "gen RTF"),
    ("stage_rtf_median.decode", "dec RTF"),
    ("stage_rtf_median.watermark", "wm RTF"),
    ("peak_rss_mb", "peak MB"),
    ("load_s", "load s"),
    ("wer_corpus", "WER"),
    ("spk_sim_mean", "spk sim"),
    ("spk_sim_min", "sim min"),
    ("other_cpu_pct_median", "other cpu%"),
    ("n_suspect", "suspect"),
    ("n_failed", "failed"),
    ("n", "n"),
]

CHILD_ENV = {
    **os.environ,
    "PYTHONUTF8": "1",
    "PYTHONWARNINGS": "ignore",
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def get(d, dotted):
    for k in dotted.split("."):
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if d != {} else None


def table(dirs):
    rows = []
    for d in [p for d in dirs for p in (sorted(glob.glob(d)) or [d])]:  # expand globs here: PowerShell does not
        p = Path(d) / "summary.json"
        if p.exists():
            rows.append((Path(d).name, json.loads(p.read_text())))
    out = ["| config | " + " | ".join(h for _, h in COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for name, s in rows:
        cells = []
        for key, _ in COLS:
            v = get(s, key)
            cells.append("" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 100 else f"{v:.0f}"))
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    md = "\n".join(out)
    print(md, flush=True)
    return md


def bench_cmd(name, cfg, defaults, **extra):
    cfg = {**defaults, **cfg, **extra}
    cfg.pop("env", None)  # per-config environment, applied by run()
    cmd = [sys.executable, str(HERE / "bench.py"), "--name", name]
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        elif v is not None and v is not False:
            cmd += [flag, str(v)]
    return cmd


def fmt_t(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?", choices=[k for k in SWEEPS])
    ap.add_argument("--limit", type=int, help="override corpus size for every config")
    ap.add_argument("--repeats", type=int, help="override repeats for every config")
    ap.add_argument("--only", nargs="*", help="subset of config names")
    ap.add_argument("--interleave", action="store_true", help="round-robin repeats across configs")
    ap.add_argument("--cooldown", type=float, default=5.0, help="seconds idle between configs")
    ap.add_argument("--table", nargs="*", help="print table for these result dirs and exit")
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()

    if args.table:
        table(args.table)
        return

    cfgs = dict(SWEEPS[args.sweep])
    if any(LIB_SLICE in str(c.get("env", {}).values()) for c in cfgs.values()) and not (Path(LIB_SLICE) / "libllama.dll").exists():
        sys.exit(f"this sweep needs the patched llama.cpp build in {LIB_SLICE} (see README: 'The patched llama.cpp')")
    defaults = cfgs.pop("_defaults", {})
    if args.limit:
        defaults["limit"] = args.limit
    if args.repeats:
        defaults["repeats"] = args.repeats
    names = args.only or list(cfgs)
    t0 = time.perf_counter()
    def _n_texts(c):
        if c.get("lines"):
            return sum(int(p.split("-")[-1]) - int(p.split("-")[0]) + 1 for p in c["lines"].split(","))
        return c.get("limit", 70)

    n_utts = {n: _n_texts({**defaults, **cfgs[n]}) * {**defaults, **cfgs[n]}.get("repeats", 3) for n in names}
    sweep_total = sum(n_utts.values())
    sweep_done = 0

    def run(cmd, k, n_rounds=None, env=None):
        tag = f"[{k}/{len(names)}]" + (f" round {n_rounds}" if n_rounds else "")
        print(f"\n=== {tag} {' '.join(cmd[2:])}  (sweep {100 * sweep_done / sweep_total:.0f}% done, elapsed {fmt_t(time.perf_counter() - t0)}) ===", flush=True)
        rc = subprocess.run(cmd, env={**CHILD_ENV, **(env or {})}).returncode
        if rc == 2:
            sys.exit("sweep stopped: preflight failed (see above). Completed configs are kept; re-run to resume.")
        if rc:
            sys.exit(f"sweep stopped: bench exited {rc}")
        time.sleep(args.cooldown)

    if not args.interleave:
        for k, n in enumerate(names, 1):
            if (Path("results") / n / "summary.json").exists():
                print(f"=== [{k}/{len(names)}] {n}: already done, skipping ===", flush=True)
            else:
                run(bench_cmd(n, cfgs[n], defaults, progress=f"{sweep_done}/{sweep_total}"), k, env={**defaults.get("env", {}), **cfgs[n].get("env", {})})
            sweep_done += n_utts[n]
    else:
        # Round r runs repeat r of every config in a fresh process, appending rows. Drift hits all configs equally.
        max_rep = max({**defaults, **cfgs[n]}.get("repeats", 3) for n in names)
        for r in range(max_rep):
            for k, n in enumerate(names, 1):
                reps = {**defaults, **cfgs[n]}.get("repeats", 3)
                if r >= reps:
                    continue
                run(bench_cmd(n, cfgs[n], defaults, repeats=1, repeat_offset=r, append=(r > 0), warmup=1, progress=f"{sweep_done}/{sweep_total}"), k, r + 1, env={**defaults.get("env", {}), **cfgs[n].get("env", {})})
                sweep_done += n_utts[n] // reps

    if not args.no_score:
        print(f"\n=== scoring {len(names)} runs (sweep elapsed {fmt_t(time.perf_counter() - t0)}) ===", flush=True)
        subprocess.run([sys.executable, str(HERE / "score.py"), *[f"results/{n}" for n in names]], check=True, env=CHILD_ENV)
        if args.sweep in SEAM_REF and SEAM_REF[args.sweep] in names:
            subprocess.run([sys.executable, str(HERE / "score.py"), "--seam-ref", f"results/{SEAM_REF[args.sweep]}", *[f"results/{n}" for n in names if n != SEAM_REF[args.sweep]]], check=True, env=CHILD_ENV)

    print(f"\n=== done in {fmt_t(time.perf_counter() - t0)} ===", flush=True)
    md = table([f"results/{n}" for n in names])
    Path(f"results/{args.sweep}_table.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
