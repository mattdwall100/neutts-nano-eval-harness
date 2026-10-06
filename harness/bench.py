"""Benchmark NeuTTS-Nano on this machine: per-stage timing, RTF, peak RSS, under controlled conditions.

Usage:
    uv run bench.py --name baseline_torch
    uv run bench.py --name q4 --backbone neuphonic/neutts-nano-q4-gguf --codec neuphonic/neucodec-onnx-decoder-int8

Writes results/<name>/rows.jsonl (one row per utterance), meta.json, summary.json and *.wav.
Score WER/similarity afterwards with score.py. Run several configs with sweep.py.

Measurement controls (see DESIGN.md):
  preflight   refuse to start if other processes use > --max-other-cpu % of CPU
  suspect     flag (never drop or re-run) any utterance whose window had other load > preflight baseline + margin
  priority    run at high process priority so background work is descheduled first
  warmup      one untimed inference before measuring
"""
import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import psutil
import soundfile as sf
import torch

SR = 24_000
FRAME_HZ = 50  # NeuCodec token rate; fixed by the model.
STAGES = ("phonemize", "prompt", "prefill", "generate", "decode", "watermark")
# "generate" is exclusive of "prefill": prefill = first backbone forward over the ~700-token prompt (reference codes
# + text); generate = the per-token decode loop. tok/s is reported against generate only, as Neuphonic's table does.


# ---------------------------------------------------------------- stage timing
class StageTimer:
    """Wrap instance methods; record (stage, start, end) intervals.

    Exclusive time per stage = own duration minus its direct children (each interval is charged
    to its smallest enclosing interval only), so a phonemize call made from inside prompt-building
    inside generate is counted once at every level.
    """

    def __init__(self):
        self.intervals: list[tuple[str, float, float]] = []
        self.n_tokens = 0
        self.prompt_tokens = 0

    def wrap(self, obj, attr, stage):
        """stage: name, or callable(*args, **kwargs) -> name decided per call."""
        fn = getattr(obj, attr)

        def wrapped(*a, **kw):
            name = stage(*a, **kw) if callable(stage) else stage
            if name is None:  # not a stage boundary (e.g. per-token decode steps stay inside "generate")
                return fn(*a, **kw)
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self.intervals.append((name, t0, time.perf_counter()))

        setattr(obj, attr, wrapped)

    def reset(self):
        self.intervals.clear()
        self.n_tokens = 0
        self.prompt_tokens = 0

    def exclusive(self) -> dict[str, float]:
        iv = self.intervals
        own = [e - s for _, s, e in iv]
        for _, s, e in iv:
            parents = [j for j, (_, s2, e2) in enumerate(iv) if s2 <= s and e <= e2 and (s2, e2) != (s, e)]
            if parents:
                own[min(parents, key=lambda j: iv[j][2] - iv[j][1])] -= e - s
        out: dict[str, float] = {}
        for (stage, _, _), d in zip(iv, own):
            out[stage] = out.get(stage, 0.0) + d
        return out


# ---------------------------------------------------------------- machine state
def peak_rss_mb() -> float:
    mi = psutil.Process().memory_info()
    return (getattr(mi, "peak_wset", None) or mi.rss) / 2**20


def power_scheme() -> str | None:
    """Windows 11 'Power mode' slider (an overlay on the classic scheme) is what throttles a laptop."""
    if sys.platform != "win32":
        return None
    overlays = {
        "ded574b5-45a0-4f42-8737-46345c09c238": "best performance",
        "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "better performance",
        "961cc777-3547-4f9d-8174-7d86181b8a7a": "best power efficiency",
        "00000000-0000-0000-0000-000000000000": "balanced",
    }
    try:
        import winreg

        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes")
        bat = psutil.sensors_battery()
        ac = bat is None or bat.power_plugged
        guid = winreg.QueryValueEx(k, f"ActiveOverlay{'Ac' if ac else 'Dc'}PowerScheme")[0]
        return f"{overlays.get(guid, guid)} (ac={ac})"
    except Exception:
        return None


def cpu_name() -> str:
    """Marketing name of the CPU; platform.processor() only gives the family/model/stepping string on Windows."""
    try:
        if sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            import subprocess

            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def pkg_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


class LoadMonitor:
    """CPU use over a window: system, own, and the top other processes by name.

    Recorded for provenance and flagged, never gated. system-minus-own carries a constant ~15 % offset on
    Windows (kernel page-ins of the mmap'd GGUF, interrupts, Defender scanning written WAVs) that is part of
    the benchmark's own footprint, so a row is only suspect when named other processes exceed the preflight
    baseline by a margin. Paired seeds + interleaving + medians are the real defence against drift.
    """

    def __init__(self, baseline_other: float | None, margin: float = 10.0):
        self.proc = psutil.Process()
        self.ncpu = psutil.cpu_count() or 1
        self.threshold = (10.0 if baseline_other is None else baseline_other) + margin  # no preflight: assume an idle desktop
        self.start()

    def _others(self):
        for p in psutil.process_iter(["name"]):
            if p.pid not in (0, self.proc.pid):
                yield p

    def start(self):
        psutil.cpu_percent(None)
        self.proc.cpu_percent(None)
        for p in self._others():
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass

    def stop(self) -> dict:
        sys_pct = psutil.cpu_percent(None)
        own_pct = self.proc.cpu_percent(None) / self.ncpu
        tops = []
        for p in self._others():
            try:
                c = p.cpu_percent(None) / self.ncpu
                if c > 0.3:
                    tops.append((round(c, 1), p.info["name"]))
            except psutil.Error:
                pass
        tops.sort(reverse=True)
        other = sum(c for c, _ in tops)
        return {
            "sys_cpu_pct": round(sys_pct, 1),
            "own_cpu_pct": round(own_pct, 1),
            "other_cpu_pct": round(other, 1),
            "other_top": [f"{n} {c}" for c, n in tops[:3]],
            "suspect": other > self.threshold,
        }


def preflight(max_other: float, seconds: float = 3.0) -> float:
    """Measure other-process CPU (attributed by process) before loading anything; refuse if egregious.

    The value is also the baseline against which per-row 'suspect' flags are judged.
    """
    own = psutil.Process()
    psutil.cpu_percent(None)
    own.cpu_percent(None)
    for p in psutil.process_iter():
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(seconds)
    psutil.cpu_percent(None)
    tops = []
    for p in psutil.process_iter(["name"]):
        try:
            if p.pid not in (0, own.pid):  # skip System Idle Process and ourselves
                tops.append((p.cpu_percent(None) / (psutil.cpu_count() or 1), p.info["name"]))
        except psutil.Error:
            pass
    tops.sort(reverse=True)
    other = sum(c for c, _ in tops)
    print(f"preflight: other processes {other:.1f}% (" + ", ".join(f"{n} {c:.1f}" for c, n in tops[:4]) + ")", flush=True)
    if other > max_other:
        print(f"PREFLIGHT FAILED: {other:.1f}% > {max_other}% before the benchmark started.")
        print("Close them and re-run, or pass --no-preflight / raise --max-other-cpu.")
        sys.exit(2)
    return round(other, 1)


def set_high_priority():
    try:
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS if sys.platform == "win32" else -10)
    except Exception as e:  # not fatal; recorded in meta
        print(f"could not raise priority: {e}")
        return False
    return True


def fmt_t(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results/<name>/")
    ap.add_argument("--backbone", default="neuphonic/neutts-nano")
    ap.add_argument("--codec", default="neuphonic/neucodec")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--ref-audio", default="samples/jo.wav")
    ap.add_argument("--ref-text", default="samples/jo.txt")
    ap.add_argument("--sim-ref", default=None, help="clip to score speaker similarity against (default: --ref-audio)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repeat-offset", type=int, default=0, help="first repeat index (interleaved sweeps)")
    ap.add_argument("--seed-base", type=int, default=1000, help="seed = seed_base + repeat; same across configs")
    ap.add_argument("--limit", type=int, default=None, help="only first N corpus lines")
    ap.add_argument("--lines", default=None, help="corpus lines to use, 1-based, e.g. 11-25,61-65 (overrides --limit; text_id = line - 1)")
    ap.add_argument("--threads", type=int, default=None, help="torch + llama.cpp decode thread count")
    ap.add_argument("--threads-batch", type=int, default=None, help="llama.cpp prefill thread count")
    ap.add_argument("--decode-cores", default=None, help="comma list of logical CPUs to pin decode steps to (e.g. 0,1,2,3 = P-cores); prefill uses all")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"], help="torch backbone dtype (reference: bf16)")
    ap.add_argument("--no-watermark", action="store_true")
    ap.add_argument("--codec-no-spin", action="store_true", help="ONNX codec: disable thread spinning (its idle workers otherwise starve the backbone after every codec call)")
    ap.add_argument("--stream", action="store_true", help="use infer_stream (GGUF only); records TTFA, stall, chunks")
    ap.add_argument("--chunk-frames", type=int, default=25, help="streaming_frames_per_chunk (50 Hz frames; 25 = 0.5 s)")
    ap.add_argument("--lookforward", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=50)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--stream-impl", default="ref", choices=["ref", "ours"], help="ref = NeuTTS.infer_stream; ours = stream.py token-level loop")
    ap.add_argument("--chunk-schedule", default=None, help="ours only: comma list of chunk sizes in frames, last repeats (e.g. 10,25,50,100)")
    ap.add_argument("--reuse-prefix", action="store_true", help="ours only: keep the KV cache between calls so the shared prompt prefix is not re-prefilled")
    ap.add_argument("--warmup", type=int, default=1, help="untimed inferences before measuring")
    ap.add_argument("--max-other-cpu", type=float, default=20.0, help="preflight refuses above this %% of other-process CPU")
    ap.add_argument("--suspect-margin", type=float, default=10.0, help="row flagged suspect if other load > preflight baseline + margin")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--append", action="store_true", help="append rows to an existing run (interleaved sweeps)")
    ap.add_argument("--progress", default=None, help="sweep-level 'done/total' utterances before this run, for the %% in the log")
    args = ap.parse_args()

    idle_cpu = None if args.no_preflight else preflight(args.max_other_cpu)
    high_prio = set_high_priority()
    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)

    from neutts import NeuTTS
    import neutts.neutts as _nn

    # NeuTTS hardcodes dtype=torch.bfloat16 for the torch backbone; override at the from_pretrained call.
    _orig_fp = _nn.AutoModelForCausalLM.from_pretrained
    _dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    _nn.AutoModelForCausalLM.from_pretrained = staticmethod(lambda *a, **kw: _orig_fp(*a, **{**kw, "dtype": _dtype}))

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # reference package prints "Loading ..." lines
        tts = NeuTTS(backbone_repo=args.backbone, codec_repo=args.codec)
    load_s = time.perf_counter() - t0
    if tts._is_quantized_model and (args.threads or args.threads_batch):
        tts.backbone._ctx.set_n_threads(args.threads or tts.backbone.n_threads, args.threads_batch or tts.backbone.n_threads_batch)
    decode_cores = [int(c) for c in args.decode_cores.split(",")] if args.decode_cores else None
    all_cores = list(range(psutil.cpu_count()))
    proc = psutil.Process()
    if args.codec_no_spin and tts._is_onnx_codec:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        tts.codec.session = ort.InferenceSession(tts.codec.session._model_path, so, providers=["CPUExecutionProvider"])
    if args.no_watermark:
        tts.watermarker = None
    if args.stream:
        if not tts._is_quantized_model:
            sys.exit("--stream needs a GGUF backbone")
        tts.streaming_frames_per_chunk = args.chunk_frames
        tts.streaming_lookforward = args.lookforward
        tts.streaming_lookback = args.lookback
        tts.streaming_overlap_frames = args.overlap
        tts.streaming_stride_samples = args.chunk_frames * tts.hop_length
    rss_after_load_mb = rss_after_load = peak_rss_mb()

    ref_text = Path(args.ref_text).read_text(encoding="utf-8").strip()
    # Reference codes are a one-off per voice; cache next to the wav (ONNX decoders cannot encode).
    ref_pt = Path(args.ref_audio).with_suffix(".pt")
    if ref_pt.exists():
        ref_codes = torch.load(ref_pt)
    else:
        ref_codes = tts.encode_reference(args.ref_audio)
        torch.save(ref_codes, ref_pt)

    timer = StageTimer()
    timer.wrap(tts, "_to_phones", "phonemize")
    timer.wrap(tts, "_apply_chat_template" if not tts._is_quantized_model else "_ggml_prompt", "prompt")
    timer.wrap(tts, "_infer_ggml" if tts._is_quantized_model else "_infer_torch", "generate")
    if decode_cores:  # restore full affinity after generation so the codec/watermark thread pools are not squeezed onto the decode cores
        _gen = getattr(tts, "_infer_ggml" if tts._is_quantized_model else "_infer_torch")

        def _gen_unpinned(*a, **kw):
            try:
                return _gen(*a, **kw)
            finally:
                proc.cpu_affinity(all_cores)

        setattr(tts, "_infer_ggml" if tts._is_quantized_model else "_infer_torch", _gen_unpinned)
    def prefill_stage(n):
        if decode_cores:  # dynamic affinity: prefill on all cores, decode steps on the chosen cores
            proc.cpu_affinity(all_cores if n > 1 else decode_cores)
        if n <= 1:
            return None  # per-token decode step: stays inside "generate"
        timer.prompt_tokens += n
        return "prefill"

    if tts._is_quantized_model:
        # llama.cpp: the first eval() of a call carries the whole prompt, later ones a single token
        timer.wrap(tts.backbone, "eval", lambda tokens, *a, **kw: prefill_stage(len(tokens)))
    else:
        timer.wrap(tts.backbone, "forward", lambda *a, **kw: prefill_stage((kw["input_ids"] if "input_ids" in kw else a[0]).shape[-1]))
    if decode_cores:  # streaming calls the codec mid-generation: give it all cores (the next backbone step re-pins itself)
        _dec = tts._decode

        def _dec_unpinned(codes):
            proc.cpu_affinity(all_cores)
            return _dec(codes)

        tts._decode = _dec_unpinned
    timer.wrap(tts, "_decode", "decode")
    if tts.watermarker is not None:
        timer.wrap(tts.watermarker, "apply_watermark", "watermark")

    _nn.print = lambda *a, **k: None  # module-level print in neutts ("Using seed N"); seed is in our row anyway

    orig_decode = tts._decode  # count generated tokens from the string handed to _decode

    def counting_decode(codes):
        timer.n_tokens = codes.count("<|speech_")
        return orig_decode(codes)

    tts._decode = counting_decode

    all_texts = [l.strip() for l in Path(args.corpus).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.lines:
        idx = [i - 1 for part in args.lines.split(",") for i in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))]
    else:
        idx = list(range(len(all_texts)))[: args.limit] if args.limit else list(range(len(all_texts)))
    texts = [all_texts[i] for i in idx]

    meta = {
        "name": args.name,
        "backbone": args.backbone,
        "codec": args.codec,
        "dtype": args.dtype if not tts._is_quantized_model else None,
        "watermark": not args.no_watermark,
        "codec_no_spin": args.codec_no_spin,
        "stream": args.stream,
        "streaming": {"impl": args.stream_impl, "chunk_frames": args.chunk_frames, "chunk_schedule": args.chunk_schedule, "lookforward": args.lookforward, "lookback": args.lookback, "overlap": args.overlap, "reuse_prefix": args.reuse_prefix} if args.stream else None,
        "threads": args.threads,
        "threads_batch": args.threads_batch,
        "decode_cores": args.decode_cores,
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
        "sim_ref_audio": args.sim_ref or args.ref_audio,
        "corpus": args.corpus,
        "limit": args.limit,
        "lines": args.lines,
        "max_other_cpu": args.max_other_cpu,
        "preflight_other_cpu_pct": idle_cpu,
        "high_priority": high_prio,
        "load_s": round(load_s, 3),
        "rss_after_load_mb": round(rss_after_load, 1),
        "llama_lib": os.environ.get("LLAMA_CPP_LIB_PATH") or "wheel",
        "output_rows": os.environ.get("LLAMA_OUTPUT_ROWS") or None,
        "cpu": cpu_name(),
        "cores": os.cpu_count(),
        "cores_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "power_scheme": power_scheme(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        **{k: pkg_version(k) for k in ("neutts", "llama-cpp-python", "onnxruntime")},
    }
    if not (args.append and (out / "meta.json").exists()):
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"loaded {args.backbone.split('/')[-1]}{'/' + args.dtype if meta['dtype'] else ''} + {args.codec.split('/')[-1]}"
        f"{'' if meta['watermark'] else ' (no watermark)'} in {load_s:.1f}s, rss {rss_after_load_mb:.0f} MB, "
        f"power '{meta['power_scheme']}', {len(texts)} texts x {args.repeats} repeats",
        flush=True,
    )

    for _ in range(args.warmup):
        tts._seed = 0
        tts.infer(texts[0], ref_codes, ref_text)
    if args.warmup:
        print(f"warmup done ({args.warmup})", flush=True)

    reps = range(args.repeat_offset, args.repeat_offset + args.repeats)
    total = len(texts) * len(reps)
    done, t_start = 0, time.perf_counter()
    sweep_done, sweep_total = map(int, args.progress.split("/")) if args.progress else (0, total)
    load = LoadMonitor(idle_cpu, args.suspect_margin)
    with open(out / "rows.jsonl", "a" if args.append else "w", encoding="utf-8") as f:
        for ti, text in zip(idx, texts):
            for rep in reps:
                seed = args.seed_base + rep
                tts._seed = seed
                timer.reset()
                load.start()
                t0 = time.perf_counter()
                if args.stream:
                    chunks, arrivals = [], []
                    if args.stream_impl == "ours":
                        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
                        from stream import stream as _stream

                        sched = tuple(int(x) for x in args.chunk_schedule.split(",")) if args.chunk_schedule else (args.chunk_frames,)
                        gen = _stream(tts, text, ref_codes, ref_text, seed, chunk_schedule=sched, lookback=args.lookback, lookforward=args.lookforward,
                                      overlap=args.overlap, watermark=not args.no_watermark, reuse_prefix=args.reuse_prefix)
                    else:
                        gen = tts.infer_stream(text, ref_codes, ref_text)
                    try:
                        for c in gen:
                            arrivals.append(time.perf_counter() - t0)
                            chunks.append(c)
                        wav = np.concatenate(chunks) if chunks else None
                    except (ValueError, AssertionError):  # shipped streaming code crashes in overlap-add when no tokens were generated
                        wav = None
                else:
                    try:
                        wav = tts.infer(text, ref_codes, ref_text)
                    except ValueError as e:  # reference raises "No valid speech tokens" when the model samples the stop token first
                        wav = None
                e2e = time.perf_counter() - t0
                sysload = load.stop()
                if wav is None or len(wav) == 0:
                    # a model failure, not a harness error: record it and carry on. Excluded from timing medians, counted in n_failed.
                    done += 1
                    f.write(json.dumps({"text_id": ti, "repeat": rep, "seed": seed, "text": text, "failed": "no speech tokens", "e2e_s": round(e2e, 3), **sysload}) + chr(10))
                    f.flush()
                    print(f"[{100 * (sweep_done + done) / sweep_total:5.1f}%] [{done}/{total}] t{ti} r{rep} FAILED: model produced no speech tokens (seed {seed})", flush=True)
                    continue

                audio_s = len(wav) / SR
                stages = timer.exclusive()
                stream_info = {}
                if args.stream:
                    # generate = residual (the generator cannot be wrapped as a stage); glue is therefore 0 by construction
                    stages["generate"] = max(e2e - sum(stages.values()), 0.0)
                    timer.n_tokens = round(len(wav) / tts.hop_length)  # _decode is called per overlapping chunk; tokens = audio frames
                    ttfa = arrivals[0]
                    cum = np.cumsum([0] + [len(c) / SR for c in chunks[:-1]])  # audio available before chunk k
                    stall = max(0.0, max(a - (ttfa + c) for a, c in zip(arrivals, cum)))
                    stream_info = {"ttfa_s": round(ttfa, 3), "n_chunks": len(chunks), "stall_s": round(stall, 3)}
                wav_name = f"t{ti:02d}_r{rep}.wav"
                sf.write(out / wav_name, wav, SR)
                row = {
                    "text_id": ti,
                    "repeat": rep,
                    "seed": seed,
                    "text": text,
                    "wav": wav_name,
                    "audio_s": round(audio_s, 3),
                    "n_tokens": timer.n_tokens,
                    "prompt_tokens": timer.prompt_tokens,
                    "prefill_tok_per_s": round(timer.prompt_tokens / stages["prefill"], 1) if stages.get("prefill") else None,
                    "e2e_s": round(e2e, 3),
                    "rtf": round(e2e / audio_s, 3),
                    "tok_per_s": round(timer.n_tokens / stages.get("generate", float("nan")), 1),
                    "stages_s": {k: round(v, 4) for k, v in stages.items()},
                    "glue_s": round(e2e - sum(stages.values()), 4),
                    "peak_rss_mb": round(peak_rss_mb(), 1),
                    **sysload,
                    **stream_info,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                done += 1
                el = time.perf_counter() - t_start
                pct = 100 * (sweep_done + done) / sweep_total
                print(
                    f"[{pct:5.1f}%] [{done}/{total}] t{ti} r{rep} audio={audio_s:5.2f}s e2e={e2e:5.2f}s rtf={row['rtf']:.3f} "
                    f"prefill={stages.get('prefill', 0):4.2f}s ({row['prompt_tokens']} tok) decode tok/s={row['tok_per_s']:6.1f} "
                    f"other={sysload['other_cpu_pct']:4.1f}%{' SUSPECT' if sysload['suspect'] else ''} "
                    + (f"ttfa={stream_info['ttfa_s']:.2f}s stall={stream_info['stall_s']:.2f}s chunks={stream_info['n_chunks']} " if stream_info else "")
                    + f"| elapsed {fmt_t(el)} eta {fmt_t(el / done * (total - done))}",
                    flush=True,
                )

    all_rows = [json.loads(l) for l in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in all_rows if not r.get("failed")]

    def med(key, sub=None):
        vals = [r[key][sub] if sub else r[key] for r in rows if (sub is None or sub in r[key])]
        return float(np.median(vals)) if vals else float("nan")

    summary = {
        "n": len(rows),
        "rtf_median": round(med("rtf"), 3),
        "rtf_p25": round(float(np.percentile([r["rtf"] for r in rows], 25)), 3),
        "rtf_p75": round(float(np.percentile([r["rtf"] for r in rows], 75)), 3),
        "e2e_s_median": round(med("e2e_s"), 3),
        "tok_per_s_median": round(med("tok_per_s"), 1),
        "prefill_tok_per_s_median": round(med("prefill_tok_per_s"), 1),
        "prompt_tokens_median": round(med("prompt_tokens")),
        "peak_rss_mb": round(max(r["peak_rss_mb"] for r in rows), 1),
        "stage_rtf_median": {k: round(med("stages_s", k) / med("audio_s"), 4) for k in STAGES},
        "glue_rtf_median": round(med("glue_s") / med("audio_s"), 4),
        "other_cpu_pct_median": round(med("other_cpu_pct"), 1),
        "other_cpu_pct_max": round(max(r["other_cpu_pct"] for r in rows), 1),
        "n_suspect": sum(1 for r in rows if r.get("suspect")),
        "n_failed": len(all_rows) - len(rows),
        **({"ttfa_median": round(med("ttfa_s"), 3), "stall_max": round(max(r["stall_s"] for r in rows), 3), "stall_median": round(med("stall_s"), 3), "n_chunks_median": round(med("n_chunks"))} if args.stream else {}),
        "measure_s": round(time.perf_counter() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps({**meta, **summary}, indent=2))
    st = summary["stage_rtf_median"]
    print((f"streaming: ttfa {summary['ttfa_median']:.2f}s stall max {summary['stall_max']:.2f}s | " if args.stream else "") +
        f"summary {args.name}: n={summary['n']} RTF {summary['rtf_median']:.2f} "
        f"(prefill {st['prefill']:.2f} + gen {st['generate']:.2f} + decode {st['decode']:.2f} + wm {st['watermark']:.3f}) "
        f"decode {summary['tok_per_s_median']:.1f} tok/s, prefill {summary['prefill_tok_per_s_median']:.0f} tok/s, "
        f"peak {summary['peak_rss_mb']:.0f} MB, other cpu {summary['other_cpu_pct_median']:.1f}% (max {summary['other_cpu_pct_max']:.1f}), "
        f"suspect {summary['n_suspect']}, {summary['measure_s']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    # self-check of the nested-interval kernel: GGUF nesting is generate > prompt > phonemize, plus prefill in generate
    _t = StageTimer()
    _t.intervals = [("phonemize", 2.0, 2.5), ("phonemize", 3.0, 3.5), ("prompt", 1.0, 4.0), ("prefill", 5.0, 7.0), ("generate", 0.0, 10.0), ("decode", 10.0, 11.0)]
    assert _t.exclusive() == {"phonemize": 1.0, "prompt": 2.0, "prefill": 2.0, "generate": 5.0, "decode": 1.0}, _t.exclusive()
    main()
